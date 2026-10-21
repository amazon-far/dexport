"""USB camera reader on a background thread (OpenCV), plus the preview helpers.

Every frame is squash-resized to the requested ``(width, height)`` (no crop: 224 x 224 is native
on no webcam, so the aspect ratio changes) and converted BGR -> RGB. The capture fourcc is part of
the observation distribution: OpenCV's own negotiation (no fourcc, the default) gives a different
color rendition than forcing MJPG (lossy, 4:2:0) or YUYV, so collection and deployment must use the
same setting; ``fourcc=`` or ``$WUJI_CAMERA_FOURCC`` ("MJPG", "YUYV"; "" / "none" / "auto" =
negotiate) overrides it. Device check: ``python -m dexport.hardware.camera --index N [--show]``.
"""

from __future__ import annotations

import argparse
import os
import threading
import time

import cv2
import numpy as np

FIRST_FRAME_TIMEOUT_S = 3.0
DARK_FRAME_MEAN = 5.0  # a frame with a lower mean pixel value is reported as nearly black


class Camera:
    """One camera. ``frame`` is the latest frame as a (H, W, 3) uint8 RGB copy, or None before
    the first one."""

    def __init__(
        self,
        index: int = 0,
        width: int = 224,
        height: int = 224,
        fps: int = 30,
        fourcc: str | None = None,
    ):
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera at index {index}")
        if fourcc is None:
            fourcc = os.environ.get("WUJI_CAMERA_FOURCC", "")
        fourcc = (fourcc or "").strip()
        if fourcc and fourcc.lower() not in ("none", "auto"):
            # before the frame size: the driver picks its mode from (fourcc, width, height)
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc[:4].upper()))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self._target_h, self._target_w = height, width

        # The negotiated mode decides the downscale (and squash) applied to every frame.
        code = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        self.requested = {
            "index": index,
            "width": width,
            "height": height,
            "fps": fps,
            "fourcc": fourcc or "auto",
        }
        self.negotiated = {
            "width": int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(self._cap.get(cv2.CAP_PROP_FPS)),
            "fourcc": "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)),
        }
        n = self.negotiated
        print(
            f"[Camera] index={index} {n['width']}x{n['height']} @ {n['fps']:.1f} FPS "
            f"fourcc={n['fourcc']} (requested {width}x{height} @ {fps})"
        )

        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    @property
    def frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def wait_for_frame(self, timeout_s: float = FIRST_FRAME_TIMEOUT_S) -> bool:
        """Wait up to ``timeout_s`` for the first frame; True once there is one. An opened device
        can still deliver nothing, and a second camera on one USB bus is slower to start."""
        deadline = time.time() + timeout_s
        while self._frame is None and time.time() < deadline:
            time.sleep(0.02)
        return self._frame is not None

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
        self._cap.release()

    def _poll_loop(self) -> None:
        while self._running:
            ok, bgr = self._cap.read()
            if not ok:
                continue
            if bgr.shape[0] != self._target_h or bgr.shape[1] != self._target_w:
                bgr = cv2.resize(bgr, (self._target_w, self._target_h))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._frame = rgb


def warn_if_dark(label: str, frame: np.ndarray | None) -> None:
    """Warn about a missing or nearly black frame (usually a wrong index or a covered lens)."""
    mean = 0.0 if frame is None else float(frame.mean())
    if mean < DARK_FRAME_MEAN:
        print(
            f"[camera] WARNING: {label} is nearly black (mean {mean:.1f}): wrong device index, "
            "or the lens is covered"
        )


def labeled_bgr(rgb: np.ndarray, label: str) -> np.ndarray:
    """An RGB frame as BGR with ``label`` written in its top-left corner, for a preview window."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(bgr, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return bgr


def hstack_tiles(tiles: list[np.ndarray]) -> np.ndarray:
    """Images side by side, each scaled to the height of the smallest one."""
    h = min(t.shape[0] for t in tiles)
    tiles = [
        t if t.shape[0] == h else cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h))
        for t in tiles
    ]
    return np.hstack(tiles)


def camera_mosaic(labeled_cams) -> np.ndarray | None:
    """The latest frame of each ``(label, camera)`` pair side by side as one labeled BGR image
    (None without cameras). A camera is anything with ``frame`` and ``requested`` (as
    :class:`Camera`); one without a frame yet is a red NO FRAME tile of its requested size, so a
    stalled camera shows rather than vanishing."""
    red, font = (0, 0, 255), cv2.FONT_HERSHEY_SIMPLEX
    tiles = []
    for label, cam in labeled_cams:
        frame = cam.frame
        if frame is None:
            h, w = int(cam.requested["height"]), int(cam.requested["width"])
            tile = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(tile, "NO FRAME", (6, h // 2), font, 0.6, red, 2, cv2.LINE_AA)
            cv2.putText(tile, label, (6, 20), font, 0.6, red, 2, cv2.LINE_AA)
            tiles.append(tile)
        else:
            tiles.append(labeled_bgr(frame, label))
    return hstack_tiles(tiles) if tiles else None


def preview_cameras(labeled_cams) -> bool:
    """Show ``(label, camera)`` pairs in one window, after a warning for each nearly black view,
    until a key is pressed or the window is closed. Returns False if 'q' was pressed (quit)."""
    cams = list(labeled_cams)
    if not cams:
        return True
    for label, cam in cams:
        warn_if_dark(f"camera {label!r}", cam.frame)
    names = ", ".join(label for label, _ in cams)
    print(f"[camera] preview ({names}): any key or closing the window continues, q quits.")
    window = "camera check (any key: continue, q: quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        while True:
            cv2.imshow(window, camera_mosaic(cams))
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                print("[camera] quit at the preview.", flush=True)
                return False
            if key != 255 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return True
    finally:
        cv2.destroyWindow(window)
        cv2.waitKey(1)


def main():
    ap = argparse.ArgumentParser(description="Check that a camera index delivers frames.")
    ap.add_argument("--index", type=int, default=0, help="OpenCV device index (/dev/videoN)")
    ap.add_argument("--width", type=int, default=224)
    ap.add_argument("--height", type=int, default=224)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--show", action="store_true", help="show the live view (q to quit)")
    args = ap.parse_args()
    cam = Camera(args.index, args.width, args.height, args.fps)
    try:
        if not cam.wait_for_frame():
            raise SystemExit(
                f"camera {args.index} delivered no frame within {FIRST_FRAME_TIMEOUT_S:.0f} s"
            )
        frame = cam.frame
        print(f"frame {frame.shape} {frame.dtype}, mean {frame.mean():.1f}")
        warn_if_dark(f"camera {args.index}", frame)
        while args.show:
            cv2.imshow(f"camera {args.index}", cv2.cvtColor(cam.frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
