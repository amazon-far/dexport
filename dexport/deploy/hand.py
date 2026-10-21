"""The deploy loop's hand: a Wuji 1 (USB) or Wuji 2 (Ethernet) hand behind the 20-D joint vector
the policy uses.

The drivers (:func:`dexport.hardware.hands.make_hand`) take a (5, 4) finger x joint grid; the
policy's 20-D vector is that grid flattened row-major (MuJoCo order: finger-major, joint-minor), so
no permutation is needed. The hand model (``wuji1`` / ``wuji2``) selects the driver.
"""

from __future__ import annotations

import time

import numpy as np

from dexport.skills.hand_models import HAND_MODELS


class WujiHand:
    """One hand. ``cfg`` is the rig's :class:`dexport.deploy.config.HandConfig` (the reset ramp);
    ``side`` is "left" or "right"; ``hand`` is the hand model."""

    def __init__(self, cfg, side: str = "left", hand: str = "wuji1"):
        self.cfg = cfg
        self.side = side
        self.transport = HAND_MODELS[hand].transport
        self._hand = None

    def connect(self) -> "WujiHand":
        # Imported here so each driver imports its own SDK only when used.
        from dexport.hardware.hands import make_hand

        name = "Wuji 2 (Ethernet)" if self.transport == "eth" else "Wuji 1 (USB)"
        print(f"[hand:{self.side}] {name}", flush=True)
        # Two Wuji 1 hands are told apart by $WUJI_SERIAL_LEFT / $WUJI_SERIAL_RIGHT, else by the
        # handedness each reports; a Wuji 2 hand is found by discovery ($WUJI_V2_ADDR_<SIDE> if
        # discovery finds none).
        self._hand = make_hand(side=self.side, transport=self.transport)
        return self

    def close(self) -> None:
        if self._hand is not None:
            try:
                self._hand.close()
            finally:
                self._hand = None

    def _driver(self):
        if self._hand is None:
            raise RuntimeError(f"the {self.side} hand is not connected")
        return self._hand

    def get_qpos_20(self) -> np.ndarray:
        """Measured joint positions, 20-D float32."""
        return np.asarray(self._driver().get_joint_position(), dtype=np.float32).reshape(20)

    def send_qpos_20(self, target_20) -> None:
        """Send absolute 20-D joint targets."""
        self._driver().send_action(np.asarray(target_20, dtype=np.float64).reshape(5, 4))

    def reset(self) -> None:
        """Ramp to the driver's open pose, then give the fingers as long again to get there: they
        trail their targets (Wuji 1 low-pass filters them at 5 Hz)."""
        self._driver().reset(
            update_rate=self.cfg.reset_update_rate, duration=self.cfg.reset_duration
        )
        time.sleep(self.cfg.reset_duration + 0.1)

    def ramp_to(self, target_20) -> None:
        """Blocking linear ramp to ``target_20`` over the reset duration (seats a start grasp
        before an episode)."""
        from dexport.hardware.hands import ramp_to

        target = np.asarray(target_20, dtype=np.float64).reshape(5, 4)
        ramp_to(self._driver(), target, self.cfg.reset_update_rate, self.cfg.reset_duration)
