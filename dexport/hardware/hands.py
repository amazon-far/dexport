"""Hand driver factory: ``make_hand(side, transport=...)`` builds the Wuji 1 USB driver
(:class:`dexport.hardware.wuji1.Hand`, ``"usb"``, the default) or the Wuji 2 Ethernet driver
(:class:`dexport.hardware.wuji2.EthernetHand`, ``"eth"``). Both expose ``send_action``,
``get_joint_position``, ``reset`` and ``close`` on a ``(5, 4)`` joint grid; each driver's SDK is
imported only when that driver is built.

Optional environment variables of the hardware drivers: ``WUJI_SERIAL_LEFT`` / ``_RIGHT`` (Wuji 1:
USB serial of that side's hand), ``WUJI_V2_ADDR_LEFT`` / ``_RIGHT`` (Wuji 2: ``ip:port`` used when
discovery finds no hand), ``WUJI_V2_READBACK_PROBE_S`` (Wuji 2: connect-time readback probe in s,
0 skips it on a command-only bench) and ``WUJI_CAMERA_FOURCC`` (camera capture format).
"""

from __future__ import annotations

import time

import numpy as np


def resolve_transport(transport: str | None = None) -> str:
    """``"usb"`` (the default, also for None) or ``"eth"``."""
    if transport is None:
        return "usb"
    if transport not in ("usb", "eth"):
        raise ValueError(f"unknown hand transport {transport!r}; expected 'usb' or 'eth'")
    return transport


def make_hand(side: str = "left", *, transport: str | None = None, **kwargs):
    """Construct and connect the hand for ``side``; ``kwargs`` go to the driver (Wuji 1:
    ``serial``, ``effort_limit_nm``; Wuji 2: ``address``, ``effort_limit_a``)."""
    if resolve_transport(transport) == "eth":
        from dexport.hardware.wuji2 import EthernetHand

        return EthernetHand(side=side, **kwargs)
    from dexport.hardware.wuji1 import Hand

    return Hand(side=side, **kwargs)


def ramp_to(hand, target: np.ndarray, update_rate: float = 100.0, duration: float = 0.5) -> None:
    """Blocking linear ramp of ``hand`` from its measured pose to ``target`` ((5, 4) rad):
    ``max(1, int(duration * update_rate))`` sends ``1 / update_rate`` s apart, so a zero
    ``duration`` still reaches the target."""
    period = 1.0 / update_rate
    steps = max(1, int(duration * update_rate))
    start = np.asarray(hand.get_joint_position(), dtype=np.float64).reshape(5, 4)
    target = np.asarray(target, dtype=np.float64).reshape(5, 4)
    for step in range(1, steps + 1):
        alpha = step / steps
        hand.send_action(start + alpha * (target - start))
        time.sleep(period)
