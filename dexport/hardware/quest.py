"""Meta Quest controller reader (``oculus_reader`` over adb): controller poses and buttons,
polled on a background thread.

``VRController.poses`` is ``{"l": 4x4, "r": 4x4}`` (the controllers seen in the last frame, in
the headset frame); ``VRController.buttons`` is a :class:`ButtonState`. A frame without poses is
ignored entirely, buttons included. ``controller_on`` is False once ``controller_timeout`` seconds
pass without a pose, and True during the first ``controller_timeout`` seconds even without data.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from dataclasses import dataclass

OCULUS_READER_INSTALL = "pip install -e '.[collect]' (the collect extra), with adb on PATH"


def import_oculus_reader():
    """``oculus_reader.reader.OculusReader``, with the install command if it is missing."""
    try:
        from oculus_reader.reader import OculusReader
    except ImportError as exc:
        raise ImportError(f"the Quest reader needs oculus_reader: {OCULUS_READER_INSTALL}") from exc
    return OculusReader


@dataclass
class ButtonState:
    """One snapshot of the controller inputs. Grips and triggers are analog in [0, 1]; the
    joysticks are (x, y) in [-1, 1] with x right and y up."""

    A: bool = False
    B: bool = False
    X: bool = False
    Y: bool = False
    left_grip: float = 0.0
    right_grip: float = 0.0
    left_trigger: float = 0.0
    right_trigger: float = 0.0
    left_joystick: tuple[float, float] = (0.0, 0.0)
    right_joystick: tuple[float, float] = (0.0, 0.0)


class VRController:
    """Polls the Quest at ``hz`` and exposes the latest poses and buttons (thread-safe)."""

    def __init__(self, hz: int = 50, controller_timeout: float = 5.0):
        self._reader = import_oculus_reader()()
        self._hz = hz
        self._controller_timeout = controller_timeout
        self._buttons = ButtonState()
        self._poses: dict = {}
        self._controller_on = False
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    @property
    def buttons(self) -> ButtonState:
        with self._lock:
            return dataclasses.replace(self._buttons)

    @property
    def poses(self) -> dict:
        with self._lock:
            return dict(self._poses)

    @property
    def controller_on(self) -> bool:
        with self._lock:
            return self._controller_on

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)

    def _poll_loop(self) -> None:
        last_read_time = time.time()
        while self._running:
            time.sleep(1.0 / self._hz)
            poses, buttons = self._reader.get_transformations_and_buttons()
            time_since_read = time.time() - last_read_time
            with self._lock:
                self._controller_on = time_since_read < self._controller_timeout
                if poses == {}:
                    continue
                self._poses = poses
                self._buttons = self._parse_buttons(buttons)
                self._controller_on = True
            last_read_time = time.time()

    @staticmethod
    def _parse_buttons(raw: dict) -> ButtonState:
        """Digital buttons are passed through; an analog value that arrives as a sequence is its
        first element, anything else is cast to float."""

        def analog(key):
            v = raw.get(key, 0.0)
            return v[0] if isinstance(v, (list, tuple)) else float(v)

        return ButtonState(
            A=raw.get("A", False),
            B=raw.get("B", False),
            X=raw.get("X", False),
            Y=raw.get("Y", False),
            left_grip=analog("leftGrip"),
            right_grip=analog("rightGrip"),
            left_trigger=analog("leftTrig"),
            right_trigger=analog("rightTrig"),
            left_joystick=VRController._parse_joystick(raw.get("leftJS")),
            right_joystick=VRController._parse_joystick(raw.get("rightJS")),
        )

    @staticmethod
    def _parse_joystick(val) -> tuple[float, float]:
        """An (x, y) float pair from a 2+ element list/tuple, else (0, 0)."""
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            return (float(val[0]), float(val[1]))
        return (0.0, 0.0)
