"""Deploy cameras: the policy's image keys and the camera behind each.

:class:`DeployCamera` reads a USB camera through :class:`dexport.hardware.camera.Camera`, the
reader data collection uses (cv2 capture, squash-resize to (W, H), BGR -> RGB, uint8): the policy
is sensitive to the image distribution, so deploy reproduces the collected pixels rather than,
say, cropping. The preview before each episode is :func:`dexport.hardware.camera.preview_cameras`.
"""

from __future__ import annotations

import numpy as np

from dexport.deploy.errors import DeployError


class DeployCamera:
    """One policy camera; ``spec`` is a :class:`dexport.collect.config.CameraConfig` (key,
    device index, width, height, fps)."""

    def __init__(self, spec):
        self.spec = spec
        self._cam = None  # dexport.hardware.camera.Camera

    def connect(self) -> "DeployCamera":
        """Open the camera and wait for its first frame."""
        from dexport.hardware.camera import FIRST_FRAME_TIMEOUT_S, Camera

        s = self.spec
        self._cam = Camera(index=s.index, width=s.width, height=s.height, fps=s.fps)
        if not self._cam.wait_for_frame(FIRST_FRAME_TIMEOUT_S):
            raise RuntimeError(
                f"camera {s.key} (index {s.index}) gave no frame in {FIRST_FRAME_TIMEOUT_S:g} s"
            )
        return self

    def close(self) -> None:
        if self._cam is not None:
            try:
                self._cam.stop()
            finally:
                self._cam = None

    @property
    def frame(self):
        """The latest frame, (H, W, 3) uint8 RGB, or None before the first one."""
        return None if self._cam is None else self._cam.frame

    @property
    def requested(self) -> dict:
        """The capture settings asked for (as the ``requested`` of a hardware ``Camera``)."""
        s = self.spec
        return {"index": s.index, "width": s.width, "height": s.height, "fps": s.fps}

    def get_rgb(self) -> np.ndarray:
        frame = self.frame
        if frame is None:
            raise RuntimeError(f"camera {self.spec.key} has no frame")
        return frame


def camera_key(im_key: str, side: str) -> str:
    """The camera behind a policy image key: ``left_rgb`` / ``right_rgb`` name themselves, and
    ``base_rgb`` (see ``dexport.minbc.data.zarr_io.CAMERA_ALIASES``) is the camera on the policy's
    hand ``side``."""
    if im_key in ("left_rgb", "right_rgb"):
        return im_key
    if im_key == "base_rgb":
        return "right_rgb" if side == "right" else "left_rgb"
    raise DeployError(f"unknown image key {im_key!r} (expected base_rgb, left_rgb or right_rgb)")
