"""Capture paired (flange pose, Quest-controller pose) samples with the hand-camera module and its
controller bracket bolted to the flange, for :mod:`dexport.calib.solve`.

``--mode probe`` checks the controller tracking without moving the arm. ``--mode jog`` drives the
flange from the terminal keyboard (Cartesian steps through the arm's IK, or single joints); Space
waits for the arm to settle, averages both poses over a 1.5 s dwell and records the pair. Tilt the
flange by >= 30 deg about several axes over ~30 captures, varying the rotations: a rotation about
an axis says nothing about the lever arm along it. The Quest headset must not move during a session.

    python -m dexport.calib.capture --mode probe --arm A
    python -m dexport.calib.capture --mode jog --arm A --out capture.npz
    python -m dexport.calib.solve --npz capture.npz          # add --hand wuji2 on a Wuji 2 rig
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from dexport.common.pose import (
    make_transform,
    mat_to_pos_quat,
    mean_transform,
    pos_quat_to_mat,
    rotation_angle_deg,
)
from dexport.common.terminal import KeyPoller
from dexport.hardware.tianji.arms import (
    ARM_SIDE,
    JOINT_LIMITS_DEG,
    add_tianji_args,
    tianji_config_from_args,
)

SIDE_KEY = {"left": "l", "right": "r"}  # the Quest controller of each side
#: An arm moving slower than this (deg/s, largest joint) counts as settled.
SETTLE_VEL_DEG_S = 0.1
#: Arrays of a capture npz, one row per capture; the npz also holds the scalar ``side``.
#: ``Q_marker`` is the controller pose (the key name is kept for existing captures).
CAPTURE_KEYS = (
    "R_flange", "Q_marker", "q_deg", "quest_pos_std", "quest_rot_std", "n_samples",
    "flange_pos_std", "n_distinct",
)  # fmt: skip


@dataclass
class JogSettings:
    """The jog's step sizes (``[`` / ``]`` halve and double the step) and the capture timing."""

    step_m: float = 0.02
    step_deg: float = 10.0  # rotation and joint step
    max_step_m: float = 0.05
    max_step_deg: float = 30.0
    dwell_s: float = 1.5  # pose averaging window
    settle_timeout_s: float = 4.0


def _pose_spread(mats, T_mean) -> tuple[np.ndarray, float]:
    """(per-axis position std (3,), rotation std in deg) of poses about their mean."""
    P = np.stack([m[:3, 3] for m in mats])
    ref = Rot.from_matrix(T_mean[:3, :3]).inv()
    angles = [np.linalg.norm((ref * Rot.from_matrix(m[:3, :3])).as_rotvec()) for m in mats]
    return P.std(axis=0), float(np.rad2deg(np.std(angles)))


def quest_dwell(vr, side_key: str, dwell_s: float, flange_fn=None) -> dict:
    """Average the controller pose (and, with ``flange_fn``, the flange pose) over one dwell.

    Returns ``T`` (the mean controller pose, None if the dwell failed), ``pos_std``, ``rot_std``,
    ``n`` (samples; negative for a frozen stream), ``n_distinct`` and ``keys_seen``; with
    ``flange_fn`` also ``T_flange`` and ``flange_pos_std``. Both poses are averaged over the same
    window, since the compliant arm creeps slightly at rest. The reader repeats its last pose
    when the adb pipe stalls, so a dwell of bit-identical samples is rejected, and
    ``n_distinct`` (distinct samples, not polls) exposes a lagging pipe.
    """
    min_samples = max(5, int(dwell_s * 10))
    mats, fmats, keys_seen, t_end = [], [], set(), time.monotonic() + dwell_s
    while time.monotonic() < t_end:
        p = vr.poses
        keys_seen.update(p.keys())
        if side_key in p:
            mats.append(np.asarray(p[side_key], dtype=np.float64).copy())
        if flange_fn is not None:
            fmats.append(np.asarray(flange_fn(), dtype=np.float64))
        time.sleep(0.005)
    out: dict = {
        "n": len(mats),
        "n_distinct": len({m[:3, 3].tobytes() for m in mats}),
        "T": None,
        "pos_std": np.full(3, np.nan),
        "rot_std": float("nan"),
        "keys_seen": sorted(keys_seen),
    }
    if fmats:
        T_f = mean_transform(fmats)
        out.update(T_flange=T_f, flange_pos_std=_pose_spread(fmats, T_f)[0])
    if len(mats) < min_samples:
        return out
    P = np.stack([m[:3, 3] for m in mats])
    if np.all(P == P[0]):
        print(f"[capture] WARNING: the controller stream is frozen ({len(P)} identical samples)")
        out["n"] = -len(mats)
        return out
    expected = max(1, int(dwell_s * 50.0))  # the reader polls at 50 Hz
    if out["n_distinct"] < 0.5 * expected:
        print(
            f"[capture] WARNING: the controller stream is slow: {out['n_distinct']} distinct "
            f"poses in {dwell_s:.1f} s (~{expected} expected)"
        )
    T = mean_transform(mats)
    ps, rs = _pose_spread(mats, T)
    out.update(T=T, pos_std=ps, rot_std=rs)
    return out


def wait_settled(get_joints, timeout_s: float = 4.0) -> float:
    """Wait until the joints (degrees, from ``get_joints()``) stop moving; returns the last
    measured speed (deg/s, largest joint)."""
    t0 = time.monotonic()
    prev, tprev, speed = np.asarray(get_joints(), dtype=np.float64), time.monotonic(), 1e9
    while time.monotonic() - t0 < timeout_s:
        time.sleep(0.05)
        now, tnow = np.asarray(get_joints(), dtype=np.float64), time.monotonic()
        speed = float(np.max(np.abs(now - prev)) / max(tnow - tprev, 1e-6))
        prev, tprev = now, tnow
        if speed <= SETTLE_VEL_DEG_S:
            return speed
    return speed


def joint_limit_violations(q_deg) -> str | None:
    """The joints of ``q_deg`` outside ``JOINT_LIMITS_DEG`` (inclusive bounds), or None."""
    bad = [
        f"J{i + 1} {v:.1f} not in [{lo}, {hi}]"
        for i, (v, (lo, hi)) in enumerate(zip(np.asarray(q_deg, float), JOINT_LIMITS_DEG))
        if not lo <= v <= hi
    ]
    return ", ".join(bad) or None


class ArmAdapter:
    """The flange-pose interface the jog loop uses, over one arm of a ``TianjiArms`` driver.

    IK does not fail loudly: an unreachable pose comes back as the closest solution, clipped to
    the joint limits. Every Cartesian step is therefore re-FK'd and refused unless it reaches the
    requested pose."""

    def __init__(self, arms, arm: str, pos_tol_m: float = 2e-3, rot_tol_deg: float = 1.0):
        self._arms, self._arm = arms, arm
        self.pos_tol_m, self.rot_tol_deg = pos_tol_m, rot_tol_deg

    def get_q(self) -> np.ndarray:
        return np.asarray(self._arms.joints_deg(self._arm), dtype=np.float64)

    def get_flange(self) -> np.ndarray:
        pos, quat = self._arms.get_ee_pose(self._arm)
        return pos_quat_to_mat(np.asarray(pos, np.float64), np.asarray(quat, np.float64))

    def goto_q(self, q) -> tuple[bool, str]:
        bad = joint_limit_violations(q)
        if bad:
            return False, f"joint limits: {bad}"
        self._arms.set_target_joints_deg(self._arm, q)  # non-blocking, watchdog off
        return True, ""

    def goto_flange(self, T) -> tuple[bool, str]:
        pos, quat = mat_to_pos_quat(T)
        q = np.asarray(self._arms.solve_ik_deg(self._arm, pos, quat), dtype=np.float64)
        bad = joint_limit_violations(q)
        if bad:
            return False, f"unreachable (joint limits: {bad})"
        pos_ik, quat_ik = self._arms.fk(self._arm, q)
        T_ik = pos_quat_to_mat(np.asarray(pos_ik, np.float64), np.asarray(quat_ik, np.float64))
        dp = float(np.linalg.norm(T_ik[:3, 3] - T[:3, 3]))
        dr = rotation_angle_deg(T_ik[:3, :3], T[:3, :3])
        if dp > self.pos_tol_m or dr > self.rot_tol_deg:
            return False, f"IK did not reach the pose ({dp * 1000:.1f} mm / {dr:.2f} deg off)"
        self._arms.set_target_joints_deg(self._arm, q)
        return True, ""

    def settle(self, timeout_s: float) -> float:
        return wait_settled(lambda: self._arms.joints_deg(self._arm), timeout_s)


JOG_HELP = """
  translation (base frame; f: flange frame)   w/s, up/down: x    a/d, left/right: y    q/e: z
  rotation about the flange's own axes        i/k: x    j/l: y    u/o: z
  joints                                      1..7: select a joint    -/=: nudge it
  SPACE  settle, dwell and record the pose    z  undo the last capture
  [ / ]  halve / double the step              h  back to the start pose    p  status
  x      save and quit                        ?  this help
"""

#: Jog keys -> (axis, sign).
JOG_TRANS_KEYS = {
    "w": (0, +1),
    "s": (0, -1),
    "up": (0, +1),
    "down": (0, -1),
    "a": (1, +1),
    "d": (1, -1),
    "left": (1, +1),
    "right": (1, -1),
    "q": (2, +1),
    "e": (2, -1),
}
JOG_ROT_KEYS = {"i": (0, +1), "k": (0, -1), "j": (1, +1), "l": (1, -1), "u": (2, +1), "o": (2, -1)}


def jog_step_target(T, key: str, step_m: float, step_deg: float, tool_frame: bool = False):
    """The flange target of one jog key from the flange pose ``T`` (None for another key).
    Translation is along the base axes (the flange's own with ``tool_frame``); rotation is always
    about the flange's own axes (a right-multiply, so the flange tilts in place)."""
    if key in JOG_TRANS_KEYS:
        ax, sgn = JOG_TRANS_KEYS[key]
        delta = np.zeros(3)
        delta[ax] = sgn * step_m
        T_new = T.copy()
        T_new[:3, 3] = T[:3, 3] + (T[:3, :3] @ delta if tool_frame else delta)
        return T_new
    if key in JOG_ROT_KEYS:
        ax, sgn = JOG_ROT_KEYS[key]
        v = np.zeros(3)
        v[ax] = np.deg2rad(sgn * step_deg)
        return T @ make_transform(np.zeros(3), Rot.from_rotvec(v).as_matrix())
    return None


def _dwell_hint(d: dict, side_key: str) -> str:
    """Why a dwell failed, and what to do."""
    if d["n"] < 0:
        return f"the stream is frozen ({-d['n']} identical samples): restart the Quest reader"
    seen = d.get("keys_seen") or []
    if d["n"] == 0 and seen and side_key not in seen:
        other = "r" if side_key == "l" else "l"
        return (
            f"the reader publishes {seen}, not '{side_key}': the poses are keyed by the physical "
            f"controller, so this rig carries the other one (use --quest-key {other})"
        )
    if d["n"] == 0:
        return (
            "no controller pose: is the Quest awake with the oculus_reader app running (the "
            "reader installs it on first connect), the controller awake and in view, and the "
            "headset listed by `adb devices`? (--mode probe)"
        )
    return f"only {d['n']} samples: tracking drops out at this pose (or raise --dwell-s)"


def jog_loop(adapter, vr, side_key: str, settings: JogSettings, *, save_cb=None) -> dict:
    """Keyboard jog and capture (``JOG_HELP``); returns the captured rows (``CAPTURE_KEYS``).

    ``save_cb(rows)`` runs after every capture and undo, so an interruption loses at most the
    current pose."""
    step_m, step_deg, tool_frame, joint = settings.step_m, settings.step_deg, False, 0
    T_start = adapter.get_flange()
    rows = {k: [] for k in CAPTURE_KEYS}

    def show(msg=""):
        print(msg, flush=True)

    def saved():
        if save_cb is not None and rows["R_flange"]:
            save_cb(rows)

    show(JOG_HELP)
    with KeyPoller() as poller:
        while True:
            batch = poller.keys()
            if not batch:
                time.sleep(0.02)
                continue
            for k in batch:
                if k == "x":
                    show(f"done: {len(rows['R_flange'])} captures")
                    return rows
                if k == "?":
                    show(JOG_HELP)
                elif k in ("[", "]"):
                    if k == "[":
                        step_m, step_deg = step_m / 2, step_deg / 2
                    else:
                        step_m = min(step_m * 2, settings.max_step_m)
                        step_deg = min(step_deg * 2, settings.max_step_deg)
                    show(f"step {step_m * 1000:.1f} mm / {step_deg:.1f} deg")
                elif k == "f":
                    tool_frame = not tool_frame
                    show(f"translation frame: {'flange' if tool_frame else 'base'}")
                elif k == "p":
                    t = np.round(adapter.get_flange()[:3, 3], 4).tolist()
                    q = np.round(adapter.get_q(), 1).tolist()
                    show(f"flange t = {t}  q = {q}  captures: {len(rows['R_flange'])}")
                elif k == "h":
                    ok, why = adapter.goto_flange(T_start)
                    show("back to the start pose" if ok else f"refused: {why}")
                elif k == "z":
                    if rows["R_flange"]:
                        for v in rows.values():
                            v.pop()
                        show(f"undone; {len(rows['R_flange'])} captures")
                        saved()
                elif k in "1234567":
                    joint = int(k) - 1
                    show(f"joint {k} selected")
                elif k in ("-", "="):
                    q = adapter.get_q().copy()
                    q[joint] += step_deg if k == "=" else -step_deg
                    ok, why = adapter.goto_q(q)
                    sign = "+" if k == "=" else "-"
                    show(f"J{joint + 1} {sign}{step_deg:.1f} deg" if ok else f"refused: {why}")
                elif k == " ":
                    _capture(adapter, vr, side_key, settings, rows, show)
                    saved()
                else:
                    T_new = jog_step_target(adapter.get_flange(), k, step_m, step_deg, tool_frame)
                    if T_new is not None:
                        ok, why = adapter.goto_flange(T_new)
                        if not ok:
                            show(f"refused: {why} (try a smaller step '[' or a joint jog)")


def _capture(adapter, vr, side_key, settings: JogSettings, rows, show) -> None:
    """Settle, dwell, and append one capture to ``rows`` (or report why not)."""
    speed = adapter.settle(settings.settle_timeout_s)
    if speed > SETTLE_VEL_DEG_S:
        show(
            f"not settled after {settings.settle_timeout_s:.0f} s ({speed:.3f} deg/s): capturing "
            "anyway"
        )
    d = quest_dwell(vr, side_key, settings.dwell_s, flange_fn=adapter.get_flange)
    if d["T"] is None:
        show(f"not captured: {_dwell_hint(d, side_key)}")
        return
    for k, v in (
        ("R_flange", d["T_flange"]),
        ("Q_marker", d["T"]),
        ("q_deg", adapter.get_q()),
        ("quest_pos_std", d["pos_std"]),
        ("quest_rot_std", d["rot_std"]),
        ("n_samples", d["n"]),
        ("flange_pos_std", d["flange_pos_std"]),
        ("n_distinct", d["n_distinct"]),
    ):
        rows[k].append(v)
    show(
        f"[{len(rows['R_flange']):3d}] captured: {d['n_distinct']} distinct controller samples, "
        f"std {np.linalg.norm(d['pos_std']) * 1000:.2f} mm, arm drift "
        f"{np.linalg.norm(d['flange_pos_std']) * 1000:.3f} mm"
    )


def save_capture(path, rows: dict, side: str) -> None:
    """Write the capture npz (``CAPTURE_KEYS`` and ``side``)."""
    arrays = {k: np.asarray(rows[k]) for k in CAPTURE_KEYS}
    np.savez(path, **arrays, side=side)


def mode_probe(args) -> None:
    """Print the tracking of both controllers every 0.2 s (no arm motion) until Ctrl-C, then a
    verdict for the rig's one."""
    from dexport.hardware.quest import VRController

    want = args.quest_key or SIDE_KEY[ARM_SIDE[args.arm]]
    vr = VRController(hz=50)
    print(f"tracking of both controllers (this rig: '{want}'); Ctrl-C to stop\n")
    print(f"{'t':>6s} | {'l |p|':>7s} {'l step':>7s} | {'r |p|':>7s} {'r step':>7s} | missed")
    last, n, n_miss, n_frozen, t0 = {}, 0, 0, 0, time.monotonic()
    try:
        while True:
            time.sleep(0.2)
            p = vr.poses
            n += 1
            cells = []
            for k in ("l", "r"):
                if k not in p:
                    cells.append(f"{'--':>7s} {'--':>7s}")
                    continue
                m = np.asarray(p[k], dtype=np.float64)
                step = np.linalg.norm(m[:3, 3] - last[k][:3, 3]) * 1000 if k in last else 0.0
                n_frozen += k == want and k in last and bool(np.all(m == last[k]))
                cells.append(f"{np.linalg.norm(m[:3, 3]):7.3f} {step:7.2f}")
                last[k] = m
            n_miss += want not in p
            print(
                f"{time.monotonic() - t0:6.1f} | {cells[0]} | {cells[1]} | {100 * n_miss / n:5.1f}%"
            )
    except KeyboardInterrupt:
        pass
    finally:
        vr.stop()
    seen = sorted(last)
    print(f"\n{n} polls, controllers seen: {seen or 'none'}; '{want}' missed {n_miss}")
    if not seen:
        print("no controller at all: wake the Quest and the controller, check `adb devices`")
    elif want not in seen:
        other = "r" if want == "l" else "l"
        print(
            f"only '{other}' publishes: this rig carries that controller (use --quest-key {other})"
        )
    elif n_miss > 0.02 * n:
        print("tracking drops out: move the headset so it sees the controller everywhere")
    elif n_frozen > 0.05 * n:
        print("the stream repeats poses: the adb pipe stalls, fix that first")
    else:
        print("tracking is healthy: continue with --mode jog")


def mode_jog(args) -> None:
    """Jog the arm and capture pairs into ``--out`` (saved after every capture and undo)."""
    from dexport.hardware.quest import VRController
    from dexport.hardware.tianji.arms import TianjiArms

    if not sys.stdin.isatty():
        raise SystemExit("--mode jog reads the keyboard: run it in a terminal")
    path = Path(args.out).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    side = ARM_SIDE[args.arm]
    side_key = args.quest_key or SIDE_KEY[side]
    settings = JogSettings(
        step_m=args.step_m,
        step_deg=args.step_deg,
        max_step_m=args.max_step_m,
        max_step_deg=args.max_step_deg,
        dwell_s=args.dwell_s,
        settle_timeout_s=args.settle_timeout_s,
    )
    vr = VRController(hz=50)
    rows = None
    try:
        with TianjiArms(tianji_config_from_args(args), (args.arm,)) as arms:
            rows = jog_loop(
                ArmAdapter(arms, args.arm),
                vr,
                side_key,
                settings,
                save_cb=lambda r: save_capture(path, r, side),
            )
    except KeyboardInterrupt:
        print("\ninterrupted: the captures so far are on disk")
    finally:
        vr.stop()
    if rows is not None and rows["R_flange"]:
        save_capture(path, rows, side)
    if not path.exists():
        raise SystemExit("no captures: nothing written")
    n = len(np.load(path)["R_flange"])
    print(
        f"{n} captures -> {path}\nnext: python -m dexport.calib.solve --npz {path}  "
        "(add --hand wuji2 on a Wuji 2 rig)"
    )


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python -m dexport.calib.capture",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    d = JogSettings()
    ap.add_argument("--mode", choices=["probe", "jog"], required=True)
    ap.add_argument("--arm", choices=["A", "B"], required=True, help="A = left, B = right")
    ap.add_argument(
        "--quest-key",
        choices=["l", "r"],
        help="the controller on the rig, if not the side's own (poses are keyed by controller)",
    )
    ap.add_argument("--out", help="capture npz (--mode jog)")
    ap.add_argument("--step-m", type=float, default=d.step_m, help="translation step (m)")
    ap.add_argument(
        "--step-deg", type=float, default=d.step_deg, help="rotation / joint step (deg)"
    )
    ap.add_argument(
        "--max-step-m", type=float, default=d.max_step_m, help="largest translation step (m)"
    )
    ap.add_argument(
        "--max-step-deg", type=float, default=d.max_step_deg, help="largest rotation step (deg)"
    )
    ap.add_argument("--dwell-s", type=float, default=d.dwell_s, help="pose averaging window (s)")
    ap.add_argument(
        "--settle-timeout-s", type=float, default=d.settle_timeout_s, help="settle wait (s)"
    )
    add_tianji_args(ap)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.mode == "probe":
        mode_probe(args)
    else:
        if args.out is None:
            raise SystemExit("--mode jog needs --out")
        mode_jog(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
