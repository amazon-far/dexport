"""Names of the channels of a recorded trajectory (``traj_<i>/<channel>`` in a zarr store).

The one definition shared by the recorder (:mod:`dexport.collect.recorder`), the channel-health
check, the training reader (:mod:`dexport.minbc.data.zarr_io`) and the deploy replay. Every
channel holds one row per recorded tick; camera channels are named by their camera key.
"""

from __future__ import annotations

from typing import NamedTuple


class SideChannels(NamedTuple):
    """The channels of one side: its hand and its Quest controller."""

    command: str  # commanded hand targets (T, 5, 4)
    measured: str  # measured hand joints (T, 5, 4)
    age: str  # age of the hand readback (T,), s
    pos: str  # controller position (T, 3)
    quat: str  # controller orientation (T, 4), xyzw quaternion


#: Recording time of each tick (T,), s.
TIMESTAMP = "timestamp"
#: The channels of the left and the right side.
CHANNELS = {
    "left": SideChannels("action", "hand_joints", "hand_joints_age", "pose_l", "quat_l"),
    "right": SideChannels("action_r", "hand_joints_r", "hand_joints_r_age", "pose_r", "quat_r"),
}
