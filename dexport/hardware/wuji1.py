"""Wuji 1 hand driver over USB (``wujihandpy >= 1.8``).

    hand = Hand("right")                    # or Hand("right", serial="<usb-serial>")
    hand.send_action(target)                # (5, 4) rad, non-blocking
    q = hand.get_joint_position()           # (5, 4) rad, non-blocking
    hand.close()

The hand runs wujihandpy's realtime controller: targets pass a 5 Hz low-pass filter and the
firmware interpolates them at 1 kHz; every joint has a 0.5 N·m torque limit.
"""

from __future__ import annotations

import atexit
import os
import threading
import time
from pathlib import Path

import numpy as np

from dexport.common.joints import WUJI1_THUMB_BASE_NEUTRAL
from dexport.hardware.hands import ramp_to

EFFORT_LIMIT_NM = 0.5
LOWPASS_CUTOFF_HZ = 5.0
_SERIAL_ENV = {"left": "WUJI_SERIAL_LEFT", "right": "WUJI_SERIAL_RIGHT"}


def _import_wujihandpy():
    try:
        import wujihandpy
    except ImportError as exc:
        raise ImportError(
            "the Wuji 1 (USB) hand driver needs wujihandpy: pip install -e '.[collect]' (or "
            "'.[deploy]')"
        ) from exc
    return wujihandpy


def _bus_serials(vid=0x0483, pid=0x2000):
    """Serial numbers of the hand-class USB devices, from sysfs (read only; [] if unreadable)."""
    out = []
    for d in sorted(Path("/sys/bus/usb/devices").glob("*")):
        try:
            if int((d / "idVendor").read_text(), 16) != vid:
                continue
            if int((d / "idProduct").read_text(), 16) != pid:
                continue
        except (OSError, ValueError):
            continue
        try:
            out.append((d / "serial").read_text().strip())
        except OSError:
            out.append("(serial unreadable)")
    return out


def read_handedness(hand) -> str:
    """``"left"`` / ``"right"`` / ``"unknown"`` for an open wujihandpy hand (object 0x5090)."""
    wujihandpy = _import_wujihandpy()
    try:
        raw = int(hand.read_handedness(timeout=2.0))
    except Exception:
        return "unknown"
    enum = getattr(getattr(getattr(wujihandpy, "_core", None), "Hand", None), "Side", None)
    left, right = (int(enum.Left), int(enum.Right)) if enum is not None else (1, 0)
    return {left: "left", right: "right"}.get(raw, "unknown")


def _open(side: str, serial):
    wujihandpy = _import_wujihandpy()
    if serial is None:
        return wujihandpy.Hand(side=side)
    hand = wujihandpy.Hand(serial_number=serial)
    actual = read_handedness(hand)
    if actual not in ("unknown", side):
        print(
            f"[Hand] WARNING: the {side} hand opened by serial {serial} reports itself as "
            f"{actual}: each hand would get the other's commands. Check the serial."
        )
    return hand


def _explain_selection(side: str, serials) -> None:
    """No hand opened as ``side`` without a serial: print the exports that select one by serial."""
    wujihandpy = _import_wujihandpy()
    print(
        f"[Hand] no hand on the bus ({len(serials)} found) could be opened as side={side!r}; "
        "is its handedness (object 0x5090) set? Select the hand by serial:"
    )
    for sn in serials:
        who = "unknown"
        try:  # a throwaway open: the caller raises right after this
            who = read_handedness(wujihandpy.Hand(serial_number=sn))
        except Exception:
            pass
        note = "   <- reports " + who.upper() if who == side else f"   ({who})"
        print(f"[Hand]   export {_SERIAL_ENV[side]}={sn}{note}")
    print(f"[Hand]   or set hand.{side}.serial in the collection config")


class Hand:
    """One Wuji 1 hand: joints enabled and the realtime controller open for the lifetime of the
    object. Thread-safe; :meth:`close` (also registered with atexit) disables the joints.

    Both hands enumerate with the same USB VID:PID, so a hand is opened by its serial number
    (``serial=``, else ``$WUJI_SERIAL_LEFT`` / ``$WUJI_SERIAL_RIGHT``) or else by the handedness the
    SDK reads from it (object 0x5090: 1 = left, 0 = right). Without a serial the open therefore
    needs the handedness flashed, also with a single hand on the bus. A hand opened by serial is
    cross-checked against its handedness. ``effort_limit_nm`` is the per-joint torque limit (None
    keeps the firmware limit).

    Never write object 0x0D:8 (soft reset; it can wedge the USB bus) or 0x5300:1, and never pass
    ``mask=`` to ``wujihandpy.Hand()`` (a masked joint fakes SDO success and reads 0).
    """

    def __init__(
        self, side: str = "left", *, serial: str | None = None, effort_limit_nm=EFFORT_LIMIT_NM
    ):
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        wujihandpy = _import_wujihandpy()
        self.side = side

        source = "serial="
        if serial is None:
            serial = (os.environ.get(_SERIAL_ENV[side]) or "").strip() or None
            source = f"${_SERIAL_ENV[side]}"
        if serial is not None:
            print(f"[Hand] opening the {side} hand by serial {serial} ({source})")
        self._hand = self._open_with_retries(serial)
        self._hand.disable_thread_safe_check()  # this class does its own locking
        self._hand.write_joint_enabled(True)
        self._closed = False
        self._lock = threading.Lock()

        # The torque limit is set before the controller streams the first target.
        if effort_limit_nm is not None:
            try:
                self._hand.write_joint_effort_limit(float(effort_limit_nm))
                print(f"[Hand] per-joint effort limit {float(effort_limit_nm):.3f} N.m")
            except Exception as exc:
                print(f"[Hand] could not set the effort limit ({type(exc).__name__}: {exc})")

        self._rt_ctx = self._hand.realtime_controller(
            enable_upstream=True, filter=wujihandpy.filter.LowPass(cutoff_freq=LOWPASS_CUTOFF_HZ)
        )
        self._controller = self._rt_ctx.__enter__()
        atexit.register(self.close)

    def _open_with_retries(self, serial):
        """Up to three opens: a hand released by another process can time out once while the
        kernel re-attaches it. An SDK argument error or an ambiguous bus fails right away."""
        for attempt in range(3):
            try:
                return _open(self.side, serial)
            except TypeError as exc:
                print(f"[Hand] wujihandpy rejected the arguments (needs wujihandpy>=1.8): {exc}")
                raise
            except Exception as exc:
                serials = _bus_serials()
                if serial is None and len(serials) > 1:
                    _explain_selection(self.side, serials)
                    raise
                if attempt == 2:
                    print(f"[Hand] wujihandpy.Hand() failed 3 times ({type(exc).__name__}: {exc})")
                    if not serials:
                        print("[Hand]   no 0483:2000 device on the USB bus: check cable and power")
                    elif serial is None:
                        _explain_selection(self.side, serials)
                    raise
                print(f"[Hand] open attempt {attempt + 1} failed ({type(exc).__name__}); retrying")
                time.sleep(1.5)

    def get_joint_position(self) -> np.ndarray:
        """Measured joint positions, (5, 4) rad (a copy)."""
        with self._lock:
            return self._controller.get_joint_actual_position().copy()

    def send_action(self, target: np.ndarray) -> None:
        """Joint position targets, (5, 4) float64 rad."""
        with self._lock:
            self._controller.set_joint_target_position(target)

    def reset(self, update_rate: float = 100.0, duration: float = 0.5) -> None:
        """Blocking ramp to the neutral pose: zeros except the thumb base (0.098 rad), whose lower
        limit is above zero."""
        target = np.zeros((5, 4), dtype=np.float64)
        target[0, 0] = WUJI1_THUMB_BASE_NEUTRAL
        ramp_to(self, target, update_rate, duration)

    def close(self) -> None:
        """Close the realtime controller and disable the joints (idempotent)."""
        if self._closed:
            return
        self._closed = True
        with self._lock:
            self._rt_ctx.__exit__(None, None, None)
        self._hand.write_joint_enabled(False)
        print(f"[Hand] joints disabled, {self.side} hand closed")
