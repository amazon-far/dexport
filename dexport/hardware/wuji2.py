"""Wuji 2 hand driver over Ethernet (``wuji-sdk``), with the same interface as the Wuji 1 driver.

    hand = EthernetHand("right")       # connects, enables, waits for the readback stream
    hand.send_action(target)           # (5, 4) rad, non-blocking
    q = hand.get_joint_position()      # (5, 4) rad, non-blocking
    hand.last_readback_age_s           # age (s) of that reading; large = a held pose
    hand.close()

Control: MIT impedance with kp 3.0 / kd 0.05 on every joint and a 0.5 A effort limit (the SDK takes
amperes, not N·m).
"""

from __future__ import annotations

import atexit
import os
import threading
import time

import numpy as np

from dexport.common.joints import WUJI1_THUMB_BASE_NEUTRAL
from dexport.hardware.hands import ramp_to

EFFORT_LIMIT_A = 0.5
_MIT_KP, _MIT_KD = 3.0, 0.05
_ENABLE_TIMEOUT_S = 5.0  # for every joint to reach ext_state Enabled after enable()
# Reader-thread rate (Hz); each iteration drains to the newest frame, so it only has to keep the
# ring from overflowing.
_READ_HZ = 60.0
# Seconds to wait at connect for the first joint_states frame; $WUJI_V2_READBACK_PROBE_S=0 skips
# the check on a bench where only commands matter.
_READBACK_PROBE_S = float(os.environ.get("WUJI_V2_READBACK_PROBE_S", "2.0"))
_STALE_WARN_S = 0.5  # warn when the readback has not refreshed for this long
_RECONNECT_STALL_S = 1.0  # re-subscribe after this long without a frame
_RECONNECT_COOLDOWN_S = 2.0  # at most one re-subscribe per this period

# Robot-LAN default addresses of the hands (static IPs, Zenoh port 7447): left .110, right .111.
# Both hands ship as .111, so the left hand must be re-IP'd to .110 before both can run.
_DEFAULT_ADDR = {"left": "192.168.1.110:7447", "right": "192.168.1.111:7447"}

_ENABLED_EXT_STATE = 2  # StatusWord.ext_state: 0 Init, 1 Ready, 2 Enabled, 3 Stopped
_N_FINGERS, _N_JOINTS = 5, 4
_N = _N_FINGERS * _N_JOINTS

# joint_states nid -> row-major grid index: finger f owns nids 5f+1 .. 5f+4 (5, 10, ... reserved).
_NID_TO_GRID_INDEX = {
    5 * f + 1 + j: f * _N_JOINTS + j for f in range(_N_FINGERS) for j in range(_N_JOINTS)
}


def _import_wuji_sdk():
    try:
        import wuji_sdk
    except ImportError as exc:
        raise ImportError(
            "the Wuji 2 (Ethernet) hand driver needs wuji_sdk: pip install -e '.[collect]' (or "
            "'.[deploy]'); the Wuji 1 USB hand uses wujihandpy instead"
        ) from exc
    return wuji_sdk


class EthernetHand:
    """One Wuji 2 hand. ``get_joint_position`` and ``send_action`` are thread-safe and do not block.

    The ``(5, 4)`` grid maps to the SDK's 20 joints row-major (thumb..pinky x joints 1-4), with no
    permutation. The hand is found by discovery and confirmed by the handedness it reports; when
    discovery finds nothing (e.g. a host firewall drops the broadcast reply) the driver connects to
    the side's static address: ``address=``, else ``$WUJI_V2_ADDR_<SIDE>``, else the robot-LAN
    default. A background thread drains the ~500 Hz joint_states stream to its newest frame, so
    reading never blocks the control loop. A hand whose stream never delivers fails at connect; a
    stream that stalls later is reported (``last_readback_age_s``, warnings) and re-subscribed.
    """

    _MAX_DRAIN = 256  # bound on recv() calls per drain (the live ring holds a few dozen frames)

    def __init__(
        self,
        side: str = "left",
        *,
        address: str | None = None,
        effort_limit_a: float = EFFORT_LIMIT_A,
    ) -> None:
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        self.side = side
        self._address = (
            address or os.environ.get(f"WUJI_V2_ADDR_{side.upper()}") or _DEFAULT_ADDR[side]
        )
        self._lock = threading.Lock()
        self._closed = False
        self._device_name = f"wuji_hand_2_{side}"
        self._last_pos = np.zeros((_N_FINGERS, _N_JOINTS), dtype=np.float64)
        self._last_recv_monotonic = 0.0  # time of the last parsed joint_states frame
        self.last_readback_age_s = 0.0
        self._last_reconnect_attempt = 0.0
        self._warn_times: dict = {}
        self._read_thread: threading.Thread | None = None
        self._read_stop = threading.Event()

        sdk = _import_wuji_sdk()
        self._sdk = sdk
        self._manager = sdk.SdkManager.instance()
        self._hand = self._connect(sdk, side)

        n_online = int(self._hand.online_joints_count().get())
        if n_online == 0:
            self._manager.disconnect_all()
            raise RuntimeError(f"[EthernetHand:{side}] 0 joints online (check power and link)")
        if n_online != _N:
            print(f"[EthernetHand:{side}] WARNING: {n_online}/{_N} joints online")
        # configure before enable
        self._hand.effort_limit().set(effort_limit_a)
        self._hand.mit_params().set((_MIT_KP, _MIT_KD))
        self._hand.enable()
        self._wait_for_enabled()
        self._sub = self._hand.joint_states().subscribe()
        self._pub = self._hand.joint_command().publish()

        # A stream that never delivers would record the connect-time snapshot as a constant
        # readback for the whole session, so refuse to start.
        try:
            self._verify_readback_stream()
        except Exception:
            self.close()
            raise
        # Seed the pose from a live frame: the first reading and reset() start from it.
        self._refresh_pose()
        self._start_read_thread()
        atexit.register(self.close)
        print(
            f"[EthernetHand:{side}] connected and enabled ({n_online} joints, "
            f"effort {effort_limit_a} A, kp/kd {_MIT_KP}/{_MIT_KD})"
        )

    def _connect(self, sdk, side: str):
        """Discovery, connect by serial and confirm the handedness; or the static address."""
        try:
            devices = self._manager.scan()
        except Exception as exc:
            print(f"[EthernetHand:{side}] scan() raised ({exc}); trying the static address")
            devices = []
        hand_devs = [d for d in devices if d.device_type == sdk.DeviceType.WujiHand2]
        if not hand_devs:
            return self._connect_by_address(side)
        for dev in hand_devs:
            hand = self._manager.connect(sn=dev.sn, device_name=self._device_name)
            reported = self._read_handedness(hand)
            if reported is None and len(hand_devs) == 1:
                print(f"[EthernetHand:{side}] handedness unreadable; using the only hand {dev.sn}")
                return hand
            if reported == side:
                print(f"[EthernetHand:{side}] matched device {dev.sn}")
                return hand
            print(f"[EthernetHand:{side}] {dev.sn} reports {reported}; skipping")
            self._manager.disconnect(device_name=self._device_name)
        raise RuntimeError(
            f"[EthernetHand:{side}] {len(hand_devs)} Wuji 2 hand(s) found, none reports {side}: "
            "set the handedness in the firmware or connect the hand alone"
        )

    def _connect_by_address(self, side: str):
        addr = self._address
        print(f"[EthernetHand:{side}] discovery found no hand; connecting to {addr}")
        try:
            hand = self._manager.connect(address=addr, device_name=self._device_name)
        except Exception as exc:
            ip = addr.split(":")[0]
            raise RuntimeError(
                f"[EthernetHand:{side}] connect(address={addr!r}) failed ({exc}). If the hand "
                f"pings, a host firewall is the usual cause: the hand answers discovery with a "
                f"broadcast that a default-drop policy discards (e.g. `sudo ufw allow from {ip}`). "
                f"The host NIC must have a static IP on the hand's subnet; the device port is "
                f"7447. Set the address with $WUJI_V2_ADDR_{side.upper()} (or "
                f"hand.{side}.address in a collection config)."
            ) from exc
        reported = self._read_handedness(hand)
        if reported is not None and reported != side:
            print(f"[EthernetHand:{side}] WARNING: the hand at {addr} reports {reported}")
        return hand

    @staticmethod
    def _read_handedness(hand) -> str | None:
        try:
            val = hand.handedness().get()
        except Exception:
            return None
        s = str(val).strip().lower() if val is not None else ""
        return "left" if "left" in s else "right" if "right" in s else None

    def _wait_for_enabled(self) -> None:
        """enable() is asynchronous and commands sent before it completes are dropped."""
        deadline = time.monotonic() + _ENABLE_TIMEOUT_S
        diag = self._hand.joint_diagnostics().subscribe()
        try:
            while time.monotonic() < deadline:
                time.sleep(0.2)
                frame = diag.recv()
                if frame is None or not frame.joints:
                    continue
                if all(e.status_word.ext_state == _ENABLED_EXT_STATE for e in frame.joints):
                    return
            raise RuntimeError(
                f"[EthernetHand:{self.side}] not all joints enabled after {_ENABLE_TIMEOUT_S} s"
            )
        finally:
            diag.close()

    def _grid_from_frame(self, frame) -> np.ndarray | None:
        """A (5, 4) grid from a joint_states frame (joints placed by nid), or None if empty."""
        if frame is None or not frame.joints:
            return None
        vec = self._last_pos.reshape(_N).copy()
        for i, e in enumerate(frame.joints):
            idx = _NID_TO_GRID_INDEX.get(e.nid, i)
            if 0 <= idx < _N and e.position is not None:
                vec[idx] = float(e.position)
        return vec.reshape(_N_FINGERS, _N_JOINTS)

    def _recv_latest(self):
        """Drain the joint_states ring and return its newest frame (None if it was empty)."""
        latest = None
        for _ in range(self._MAX_DRAIN):
            try:
                frame = self._sub.recv()
            except Exception as exc:
                self._warn_throttled(
                    "recv_error",
                    f"[EthernetHand:{self.side}] joint_states recv() raised "
                    f"({type(exc).__name__}: {exc}); holding the last pose",
                )
                break
            if frame is None:
                break
            latest = frame
        return latest

    def _store_pose(self, grid: np.ndarray) -> None:
        self._last_pos = grid
        self._last_recv_monotonic = time.monotonic()

    def _refresh_pose(self) -> None:
        grid = self._grid_from_frame(self._recv_latest())
        if grid is not None:
            self._store_pose(grid)

    def _warn_throttled(self, key: str, msg: str, period_s: float = 2.0) -> None:
        now = time.monotonic()
        if now - self._warn_times.get(key, -1e9) >= period_s:
            self._warn_times[key] = now
            print(msg, flush=True)

    def _attempt_resubscribe(self) -> None:
        """Replace a stalled joint_states subscription (rate limited; never raises). Called by the
        reader thread with the lock held."""
        now = time.monotonic()
        if now - self._last_reconnect_attempt < _RECONNECT_COOLDOWN_S:
            return
        self._last_reconnect_attempt = now
        print(
            f"[EthernetHand:{self.side}] WARNING: no joint_states for "
            f"{now - self._last_recv_monotonic:.1f}s: re-subscribing (the recorded readback is a "
            "held pose until it recovers)",
            flush=True,
        )
        try:
            old, self._sub = self._sub, self._hand.joint_states().subscribe()
            try:
                old.close()
            except Exception:
                pass
            grid = self._grid_from_frame(self._recv_latest())
            if grid is not None:
                self._store_pose(grid)
                print(f"[EthernetHand:{self.side}] joint_states live again", flush=True)
            else:
                print(
                    f"[EthernetHand:{self.side}] WARNING: still no frame; retrying every "
                    f"{_RECONNECT_COOLDOWN_S:.0f}s",
                    flush=True,
                )
        except Exception as exc:
            print(
                f"[EthernetHand:{self.side}] WARNING: re-subscribe failed "
                f"({type(exc).__name__}: {exc})",
                flush=True,
            )

    def _watch_staleness(self) -> None:
        """With the lock held: warn about a stalled readback and re-subscribe if needed."""
        age = time.monotonic() - self._last_recv_monotonic
        if age >= _STALE_WARN_S:
            self._warn_throttled(
                "stale",
                f"[EthernetHand:{self.side}] joint_states STALE: no fresh frame in {age:.2f}s "
                "(the readback is a held pose)",
            )
        if age >= _RECONNECT_STALL_S:
            self._attempt_resubscribe()

    def _verify_readback_stream(self) -> None:
        """Raise if no joint_states frame arrives within ``_READBACK_PROBE_S`` (0 skips it)."""
        timeout_s = _READBACK_PROBE_S
        if timeout_s <= 0:
            return
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            frame = self._recv_latest()
            if frame is not None and getattr(frame, "joints", None):
                return
            time.sleep(0.02)
        raise RuntimeError(
            f"[EthernetHand:{self.side}] readback stream DEAD: no joint_states frame within "
            f"{timeout_s:.1f}s although the hand accepts commands. Usually a duplicate IP: both "
            f"hands ship as 192.168.1.111, so give the {self.side} hand its own address. "
            "WUJI_V2_READBACK_PROBE_S=0 skips this check when only commands matter."
        )

    def _read_once(self) -> None:
        """One reader iteration: store the newest frame, then run the staleness watchdog."""
        grid = self._grid_from_frame(self._recv_latest())
        with self._lock:
            if grid is not None:
                self._store_pose(grid)
            self._watch_staleness()

    def _start_read_thread(self) -> None:
        """Make the reader thread the only caller of recv(), at ``_READ_HZ``: it keeps the pose
        cache fresh and runs the staleness watchdog, so get_joint_position is a cache read."""
        dt = 1.0 / _READ_HZ

        def _loop():
            t0, n = time.monotonic(), 0
            while not self._read_stop.is_set():
                try:
                    self._read_once()
                except Exception:
                    pass
                n += 1
                wait = (t0 + dt * n) - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                else:  # fell behind: re-anchor instead of busy-spinning
                    t0, n = time.monotonic(), 0

        self._read_thread = threading.Thread(
            target=_loop, name=f"wuji2_read_{self.side}", daemon=True
        )
        self._read_thread.start()

    def get_joint_position(self) -> np.ndarray:
        """Measured joint positions, (5, 4) rad; sets ``last_readback_age_s``."""
        with self._lock:
            self.last_readback_age_s = time.monotonic() - self._last_recv_monotonic
            return self._last_pos.copy()

    def send_action(self, target: np.ndarray) -> None:
        """Joint position targets, (5, 4) float64 rad (not clamped, as on Wuji 1)."""
        vec = np.asarray(target, dtype=np.float64).reshape(_N)
        with self._lock:
            self._pub.send([self._sdk.JointCommand(float(p), 0.0, 0.0) for p in vec])

    def reset(self, update_rate: float = 100.0, duration: float = 0.5) -> None:
        """Blocking ramp to the start pose of the Wuji 1 driver (zeros, thumb base 0.098 rad), kept
        as collected and harmless on Wuji 2, whose thumb base admits 0. The collector then eases
        the hand to the Wuji 2 neutral (zeros) before recording; deploy does not."""
        target = np.zeros((_N_FINGERS, _N_JOINTS), dtype=np.float64)
        target[0, 0] = WUJI1_THUMB_BASE_NEUTRAL
        ramp_to(self, target, update_rate, duration)

    def close(self) -> None:
        """Disable the joints and disconnect this hand only (idempotent)."""
        if self._closed:
            return
        self._closed = True
        if self._read_thread is not None:  # stop the reader before closing its subscription
            self._read_stop.set()
            self._read_thread.join(timeout=1.0)
        with self._lock:
            for step in (
                lambda: self._pub.close(),
                lambda: self._sub.close(),
                lambda: self._hand.disable(),
                lambda: self._manager.disconnect(device_name=self._device_name),
            ):
                try:
                    step()
                except Exception as exc:
                    print(f"[EthernetHand:{self.side}] close step failed: {exc}")
        print(f"[EthernetHand:{self.side}] joints disabled, hand closed")
