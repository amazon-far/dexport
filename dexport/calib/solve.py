"""Fit the Quest-controller -> flange mount ``M`` from paired poses.

Each capture gives the flange pose ``R_i`` and the controller pose ``Q_i``, with ``Q_i = X R_i M``
(``X``: the unknown, fixed robot-base -> Quest-world transform). Two captures of one session cancel
``X``: ``R_i^-1 R_j = M (Q_i^-1 Q_j) M^-1``, the conjugation deploy applies to every action. A
session is solved in closed form (Park-Martin) and refined by Levenberg-Marquardt; sessions of one
side are pooled through their within-session pairs, and a pooled fit is the closed form only (the
Quest world frame differs between sessions, so the absolute-pose refinement does not apply). The fit
is printed as the config lines of ``dexport.deploy.config`` (its as-calibrated convention).

    python -m dexport.calib.solve --npz capture.npz [--hand wuji2]
    python -m dexport.calib.solve --npz a.npz b.npz c.npz        # pooled sessions of one side
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

from dexport.common.pose import invert_transform, make_transform, mean_transform, rotation_angle_deg
from dexport.deploy.config import HAND_TO_FLANGE_R, hand_to_flange_t, mount_transform

#: Pairs whose flange rotates by less than this (deg) say almost nothing about the lever arm.
MIN_PAIR_ROT_DEG = 10.0
#: Arrays the solver reads from a capture npz (written by ``dexport.calib.capture``).
REQUIRED_KEYS = ("R_flange", "Q_marker", "quest_pos_std", "quest_rot_std")


def pose_err(Ta: np.ndarray, Tb: np.ndarray) -> tuple[float, float]:
    """(translation mm, rotation deg) between two poses."""
    dp = float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]) * 1000.0)
    return dp, rotation_angle_deg(Ta[:3, :3], Tb[:3, :3])


def _load_capture(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    missing = [k for k in REQUIRED_KEYS if k not in d]
    if missing:
        raise SystemExit(
            f"{npz_path}: not a capture of dexport.calib.capture (no {', '.join(missing)})"
        )
    return d


def _capture_side(npz_path) -> str:
    d = np.load(npz_path, allow_pickle=True)
    if "side" not in d:
        raise SystemExit(f"{npz_path}: the capture has no 'side'; pass --side")
    return str(d["side"])


def load_session(npz_path, min_poses: int = 4):
    """One capture session as ``(R_list, Q_list, sig_t, sig_r)``: the flange and controller poses
    of the usable captures (a non-finite pose is dropped) and the median dwell noise of the
    controller (m, deg)."""
    d = _load_capture(npz_path)
    R_list = [np.asarray(m, dtype=np.float64) for m in d["R_flange"]]
    Q_list = [np.asarray(m, dtype=np.float64) for m in d["Q_marker"]]
    sig_t = float(np.nanmedian(d["quest_pos_std"]))
    sig_r = float(np.nanmedian(d["quest_rot_std"]))
    good = np.array(
        [np.all(np.isfinite(r)) and np.all(np.isfinite(q)) for r, q in zip(R_list, Q_list)]
    )
    if not np.all(good):
        print(f"{npz_path}: dropping {int((~good).sum())} of {len(good)} non-finite captures")
        R_list = [r for r, g in zip(R_list, good) if g]
        Q_list = [q for q, g in zip(Q_list, good) if g]
    if len(R_list) < min_poses:
        raise SystemExit(f"{npz_path}: only {len(R_list)} usable captures, nothing to fit")
    return R_list, Q_list, sig_t, sig_r


def build_pairs(R_list, Q_list, min_rot_deg: float = MIN_PAIR_ROT_DEG):
    """Every capture pair ``i < j`` whose flange rotates by at least ``min_rot_deg``:
    ``(A_ee, A_q, idx)`` with ``A_ee = R_i^-1 R_j`` and ``A_q = Q_i^-1 Q_j``, so that
    ``A_ee = M A_q M^-1``."""
    A_ee, A_q, idx = [], [], []
    for i in range(len(R_list)):
        Ri_inv, Qi_inv = invert_transform(R_list[i]), invert_transform(Q_list[i])
        for j in range(i + 1, len(R_list)):
            aee = Ri_inv @ R_list[j]
            if rotation_angle_deg(np.eye(3), aee[:3, :3]) < min_rot_deg:
                continue
            A_ee.append(aee)
            A_q.append(Qi_inv @ Q_list[j])
            idx.append((i, j))
    if not A_ee:
        raise SystemExit(f"no capture pair rotates by >= {min_rot_deg} deg: nothing to fit")
    return np.stack(A_ee), np.stack(A_q), np.asarray(idx)


def fit_t_linear(A_ee: np.ndarray, A_q: np.ndarray, R_M: np.ndarray) -> np.ndarray:
    """The mount translation for the rotation ``R_M``: least squares of
    ``(I - R_ee) t = p_ee - R_M p_q`` over the pairs. Only rotation reveals the lever arm, and
    a rotation about an axis says nothing about it along that axis, so the captures must tilt the
    flange about several axes (>= 30 deg); the residual cannot reveal a session that did not."""
    lhs = (np.eye(3)[None] - A_ee[:, :3, :3]).reshape(-1, 3)
    rhs = (A_ee[:, :3, 3] - A_q[:, :3, 3] @ R_M.T).reshape(-1)
    return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def fit_park_martin(A_ee: np.ndarray, A_q: np.ndarray) -> np.ndarray:
    """Closed-form M (Park & Martin): the rotation maps the pairs' controller rotation vectors
    onto their flange rotation vectors (Kabsch), the translation is :func:`fit_t_linear`."""
    a = np.stack([Rot.from_matrix(x[:3, :3]).as_rotvec() for x in A_ee])
    b = np.stack([Rot.from_matrix(x[:3, :3]).as_rotvec() for x in A_q])
    U, _, Vt = np.linalg.svd(a.T @ b)
    R_M = U @ np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))]) @ Vt
    return make_transform(fit_t_linear(A_ee, A_q, R_M), R_M)


def recover_X(R_list, Q_list, M) -> np.ndarray:
    """``X`` (robot base -> Quest world) given M: the mean of ``Q_i M^-1 R_i^-1``."""
    M_inv = invert_transform(M)
    return mean_transform([q @ M_inv @ invert_transform(r) for q, r in zip(Q_list, R_list)])


def _pack(X, M) -> np.ndarray:
    return np.concatenate(
        [
            Rot.from_matrix(X[:3, :3]).as_rotvec(),
            X[:3, 3],
            Rot.from_matrix(M[:3, :3]).as_rotvec(),
            M[:3, 3],
        ]
    )


def _unpack(p):
    return (
        make_transform(p[3:6], Rot.from_rotvec(p[0:3]).as_matrix()),
        make_transform(p[9:12], Rot.from_rotvec(p[6:9]).as_matrix()),
    )


def fit_refine(R_list, Q_list, X0, M0, sig_t_m: float, sig_r_deg: float):
    """Levenberg-Marquardt over ``(X, M)`` on the absolute residuals ``Q_i`` vs ``X R_i M``
    (position, and rotation vector of ``(X R_i M)^-1 Q_i``), whitened by the controller's dwell
    noise. Returns ``(X, M)``."""
    sig_r = np.deg2rad(max(sig_r_deg, 1e-3))
    sig_t = max(sig_t_m, 1e-6)

    def residual(p):
        X, M = _unpack(p)
        out = []
        for q, r in zip(Q_list, R_list):
            P = X @ r @ M
            out.append((q[:3, 3] - P[:3, 3]) / sig_t)
            out.append(Rot.from_matrix(P[:3, :3].T @ q[:3, :3]).as_rotvec() / sig_r)
        return np.concatenate(out)

    sol = least_squares(residual, _pack(X0, M0), method="lm", xtol=1e-15, ftol=1e-15)
    return _unpack(sol.x)


def residual_stats(R_list, Q_list, X, M) -> tuple[float, float]:
    """RMS (mm, deg) of ``Q_i`` vs ``X R_i M``."""
    et, er = [], []
    for q, r in zip(Q_list, R_list):
        a, b = pose_err(X @ r @ M, q)
        et.append(a)
        er.append(b)
    return float(np.sqrt(np.mean(np.square(et)))), float(np.sqrt(np.mean(np.square(er))))


def pair_residual(A_ee, A_q, M) -> tuple[float, float]:
    """RMS (mm, deg) of ``A_ee`` vs ``M A_q M^-1``: the residual of pooled sessions, which share
    no Quest world frame."""
    M_inv = invert_transform(M)
    et, er = [], []
    for ae, aq in zip(A_ee, A_q):
        pred = M @ aq @ M_inv
        et.append(np.linalg.norm(ae[:3, 3] - pred[:3, 3]))
        er.append(rotation_angle_deg(ae[:3, :3], pred[:3, :3]))
    return float(np.sqrt(np.mean(np.square(et))) * 1000.0), float(np.sqrt(np.mean(np.square(er))))


def decompose_as_calibrated(M_measured: np.ndarray) -> np.ndarray:
    """``T_quest_to_hand`` of a measured mount in the config's as-calibrated convention: factored
    against the standoff with the spacer (the capture setup), so ``mount_transform(hand, side,
    spacer=True)`` reproduces ``M_measured`` and ``spacer=False`` removes the spacer."""
    T_h2f = make_transform(hand_to_flange_t(True), HAND_TO_FLANGE_R)
    return invert_transform(T_h2f) @ M_measured


def _paste(M_measured, side: str, hand: str) -> dict:
    """Print the config lines of the fit and the mount deploy then uses with and without
    ``--spacer``; returns the JSON fields of the paste."""
    T = decompose_as_calibrated(M_measured)
    sfx = "_V2" if hand == "wuji2" else ""
    lr = "RH" if side == "right" else "LH"
    name_t, name_r = f"QUEST_TO_{lr}_T{sfx}", f"QUEST_TO_{lr}_R{sfx}"
    t = ", ".join(f"{v:.6f}" for v in T[:3, 3])
    print(f"\npaste into dexport/deploy/config.py ({hand}, {side}):\n")
    print(f"{name_t}: np.ndarray = np.array([{t}], dtype=np.float64)")
    print(f"{name_r}: np.ndarray = _orthonormalize(\n    np.array(\n        [")
    for row in T[:3, :3]:
        print(f"            [{', '.join(f'{v:+.9f}' for v in row)}],")
    print("        ],\n        dtype=np.float64,\n    )\n)\n")
    for label, spacer in (("with --spacer", True), ("without --spacer", False)):
        M = make_transform(hand_to_flange_t(spacer), HAND_TO_FLANGE_R) @ T
        print(f"deploy {label:16s} uses M.t = {np.round(M[:3, 3] * 1000.0, 2).tolist()} mm")
    return {
        "quest_to_hand_t_as_calibrated": T[:3, 3].tolist(),
        "quest_to_hand_R_as_calibrated": T[:3, :3].tolist(),
        "paste_constants": {"t": name_t, "R": name_r},
        "hand": hand,
    }


def _report(M, side, hand, res_mm, res_deg) -> dict:
    """Print the fit and the paste; returns the JSON fields shared by sessions and pools."""
    dt, dr = pose_err(M, mount_transform(hand, side, spacer=True))
    print(f"residual {res_mm:.2f} mm / {res_deg:.3f} deg RMS")
    print(
        f"M.t = {np.round(M[:3, 3] * 1000.0, 2).tolist()} mm, {dt:.1f} mm / {dr:.2f} deg from the "
        f"deployed {hand} {side} mount"
    )
    return {
        "M_measured": M.tolist(),
        **_paste(M, side, hand),
        "residual_mm": res_mm,
        "residual_deg": res_deg,
    }


def solve_session(npz, *, side, hand="wuji1", min_rot=MIN_PAIR_ROT_DEG) -> dict:
    """One session: Park-Martin, then the Levenberg-Marquardt refinement over ``(X, M)``; the
    residual is that of the absolute poses. Returns the JSON record."""
    R_list, Q_list, sig_t, sig_r = load_session(npz)
    A_ee, A_q, _ = build_pairs(R_list, Q_list, min_rot)
    n = len(R_list)
    print(f"{npz}: {n} captures, {len(A_ee)} of {n * (n - 1) // 2} pairs rotate >= {min_rot:g} deg")
    print(f"controller dwell noise {sig_t * 1000:.2f} mm / {sig_r:.3f} deg (median)")
    M0 = fit_park_martin(A_ee, A_q)
    X, M = fit_refine(R_list, Q_list, recover_X(R_list, Q_list, M0), M0, sig_t, sig_r)
    dt, dr = pose_err(M0, M)
    print(f"closed form -> refined: {dt:.2f} mm / {dr:.3f} deg")
    res = residual_stats(R_list, Q_list, X, M)
    return {
        "source_npz": str(npz),
        "side": side,
        "n_poses_used": n,
        **_report(M, side, hand, *res),
    }


def solve_pooled(npzs, *, side, hand="wuji1", min_rot=MIN_PAIR_ROT_DEG) -> dict:
    """Several sessions of one side: Park-Martin on their stacked within-session pairs (the Quest
    world frame resets between sessions, so a pair never spans two); the residual is the pairs'.
    Per session: its own closed-form translation and the pooled mount's residual on its pairs.
    Returns the JSON record."""
    sessions = []
    for p in npzs:
        R_list, Q_list, _, _ = load_session(p)
        A_ee, A_q, _ = build_pairs(R_list, Q_list, min_rot)
        sessions.append((len(R_list), A_ee, A_q))
        print(f"{p}: {len(R_list)} captures, {len(A_ee)} pairs rotate >= {min_rot:g} deg")
    A_ee = np.concatenate([s[1] for s in sessions])
    A_q = np.concatenate([s[2] for s in sessions])
    M = fit_park_martin(A_ee, A_q)
    Ms = [fit_park_martin(a, b) for _, a, b in sessions]
    res = [pair_residual(a, b, M) for _, a, b in sessions]
    pairs = [(i, j) for i in range(len(Ms)) for j in range(i + 1, len(Ms))]
    dts = [np.linalg.norm(Ms[i][:3, 3] - Ms[j][:3, 3]) * 1000.0 for i, j in pairs]
    drs = [rotation_angle_deg(Ms[i][:3, :3], Ms[j][:3, :3]) for i, j in pairs]
    print("per session: its own closed-form M.t (mm); the pooled mount's pair residual")
    for p, Mi, (rm, rd) in zip(npzs, Ms, res):
        t_mm = np.round(Mi[:3, 3] * 1000.0, 2).tolist()
        print(f"  {Path(p).name:24s} {t_mm}  {rm:.2f} mm / {rd:.2f} deg")
    if dts:
        print(f"largest disagreement between sessions: {max(dts):.2f} mm / {max(drs):.2f} deg")
    return {
        "source_npz": [str(p) for p in npzs],
        "side": side,
        "n_sessions": len(sessions),
        "n_poses": sum(s[0] for s in sessions),
        "n_pairs": len(A_ee),
        **_report(M, side, hand, *pair_residual(A_ee, A_q, M)),
        "per_session_M_t_mm": [(m[:3, 3] * 1000.0).tolist() for m in Ms],
        "per_session_spread_t_mm": max(dts) if dts else 0.0,
        "per_session_spread_rot_deg": max(drs) if drs else 0.0,
        "per_session_residual_mm": [r[0] for r in res],
        "per_session_residual_deg": [r[1] for r in res],
    }


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python -m dexport.calib.solve",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--npz",
        nargs="+",
        required=True,
        help="capture npz from dexport.calib.capture; several files of one side are pooled",
    )
    ap.add_argument(
        "--side", choices=["left", "right"], help="override the side stored in the capture"
    )
    ap.add_argument(
        "--hand", choices=["wuji1", "wuji2"], default="wuji1", help="hand model of the rig"
    )
    ap.add_argument(
        "--min-pair-rot",
        type=float,
        default=MIN_PAIR_ROT_DEG,
        help=f"least flange rotation of a pair, deg (default {MIN_PAIR_ROT_DEG:g})",
    )
    ap.add_argument(
        "--out",
        help="result JSON (default: <npz stem>_solution.json next to the npz, or "
        "pooled_<side>_solution.json for several; 'none' writes nothing)",
    )
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    paths = [Path(p).expanduser().resolve() for p in args.npz]
    for p in paths:
        if not p.exists():
            raise SystemExit(f"no such file: {p}")
    side = args.side
    if side is None:
        sides = {_capture_side(p) for p in paths}
        if len(sides) > 1:
            raise SystemExit(f"mixed sides {sorted(sides)}: pool each side separately")
        side = sides.pop()
    kw = dict(side=side, hand=args.hand, min_rot=args.min_pair_rot)
    if len(paths) == 1:
        rec = solve_session(paths[0], **kw)
        out = paths[0].with_name(paths[0].stem + "_solution.json")
    else:
        rec = solve_pooled(paths, **kw)
        out = paths[0].with_name(f"pooled_{side}_solution.json")
    if args.out is not None:
        out = None if args.out.lower() == "none" else Path(args.out).expanduser().resolve()
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=2))
        print(f"\nresult -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
