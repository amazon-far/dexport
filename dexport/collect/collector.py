"""Demonstration collector: ``Collector(cfg).run()``.

One collector serves Wuji 1 or Wuji 2, one or two hands and any number of cameras. The Quest
reader, the cameras and the Wuji 2 readback run on background threads. The input loop turns Quest
buttons and sticks into each hand's command (:mod:`dexport.skills.controller`); the record loop
runs at ``hz`` (30 Hz), sends the latest commands, reads the hands back and records one row of
every channel (:mod:`dexport.collect.recorder`), so it is the single recording loop.
"""

from __future__ import annotations

import sys
import threading
import time

import cv2
import numpy as np

from dexport.collect.config import CollectConfig
from dexport.collect.recorder import EpisodeRecorder
from dexport.hardware.camera import Camera, camera_mosaic, preview_cameras, warn_if_dark
from dexport.hardware.hands import make_hand
from dexport.hardware.quest import ButtonState, VRController
from dexport.skills.controller import SkillController
from dexport.skills.hand_models import GRASP_DESC
from dexport.skills.rotation.contract import POLICY_HZ

SIDES = ("left", "right")
HAND_ACTION_BUTTON = {"left": "X", "right": "A"}
GRIP_THRESHOLD = 0.5
RECORD_BUTTON = "B"
DISCARD_BUTTON = "Y"
RECORD_TOGGLE_KEY = "x"
DISCARD_KEY = "d"
QUIT_KEY = "q"
VIEWER_WINDOW = "dexport collect"


class Collector:
    """Owns the devices, the per-hand skill controllers and the recorder of one session.

    B and Y (and their keyboard backups) act on the whole session; X / A, the grip, the index
    trigger and the stick act on the hand of their own side:

      B (or x + Enter)  start / stop recording: a start first resets the hands and begins once
                        they have settled; a stop saves the episode, then resets the hands
      Y (or d + Enter)  discard the current episode
      X / A             thumb <-> finger mode; in screw mode: start, freeze, back to the start grasp
      grip              enter / leave screw mode (Wuji 1 with a ``rotation_skill:`` block)
      stick             thumb mode: jog the thumb; finger mode: down closes, up opens the grasp
      index trigger     finger mode: next grasp
      q + Enter         quit

    The keyboard backups are consumed by the input loop, which pauses while the Quest sends no
    poses; q works at any time.
    """

    def __init__(self, cfg: CollectConfig):
        self.cfg = cfg
        self.hand_model = cfg.hand_model
        self._hz = cfg.hz
        # Before any device, so a bad checkpoint or an occupied session directory fails fast.
        runners = self._load_rotation_runners()
        traj = cfg.trajectory
        self._recorder = EpisodeRecorder(
            traj.save_dir,
            tag=traj.tag,
            image_keys=[c.key for c in cfg.cameras],
            save_pose_l=traj.save_pose_l,
            save_quat_l=traj.save_quat_l,
            save_pose_r=traj.save_pose_r,
            save_quat_r=traj.save_quat_r,
            save_readback_age=self.hand_model.records_readback_age,
        )
        self._vr = VRController(hz=self._hz)
        self._hands = self._open_hands()
        self._cameras = self._open_cameras()
        self._skills = {
            side: SkillController(
                self.hand_model, self._hz, cfg.skills, side=side, rotation=runners.get(side)
            )
            for side in SIDES
        }

        neutral = self.hand_model.neutral_pose()
        self._lock = threading.Lock()  # guards the commands and readbacks shared by the loops
        self._commands = {side: neutral.copy() for side in SIDES}
        self._meas = {side: neutral.copy() for side in SIDES}
        self._pending_start = False  # a start waiting for the hands to finish their reset
        self._record_toggle_requested = False
        self._discard_requested = False
        self._action_btn_was_pressed = {side: False for side in SIDES}
        self._grip_was_pressed = {side: False for side in SIDES}
        self._record_btn_was_pressed = False
        self._discard_btn_was_pressed = False
        self._running = False
        self._viewer_open = False
        self._viewer_sized = False

    def _load_rotation_runners(self) -> dict:
        """One rotation runner per side (the actor is shared), or {} without a ``rotation_skill:``
        block. A grip on either side can enter screw mode, whether that hand is connected or not."""
        if self.cfg.rotation_skill is None:
            return {}
        path = self.cfg.rotation_skill.ckpt
        if not path.is_file():
            raise SystemExit(
                f"[collect] rotation-skill checkpoint not found: {path} "
                "(rotation_skill.ckpt; see checkpoints/rotation_skill/README.md)"
            )
        from dexport.skills.rotation.actor import load_rotation_actor
        from dexport.skills.rotation.runner import RotationRunner

        try:
            actor = load_rotation_actor(path)
            runners = {side: RotationRunner(actor) for side in SIDES}
        except Exception as e:
            raise SystemExit(f"[collect] cannot use the rotation skill in {path}: {e}") from e
        root = "the checkpoint" if runners["left"].root_from_ckpt else "the contract default"
        print(
            f"[collect] rotation skill {path} ({POLICY_HZ:.0f} Hz, hand root pose from {root}); "
            "the grip enters screw mode"
        )
        return runners

    def _open_hands(self) -> dict:
        hands = {}
        for side in SIDES:
            side_cfg = getattr(self.cfg.hand, side)
            if not side_cfg.enabled:
                continue
            kwargs = {}
            if side_cfg.serial is not None:
                kwargs["serial"] = side_cfg.serial
            if side_cfg.address is not None:
                kwargs["address"] = side_cfg.address
            hands[side] = make_hand(side, transport=self.hand_model.transport, **kwargs)
        return hands

    def _open_cameras(self) -> dict:
        """Open every configured camera and wait for its first frame (an opened device can still
        be dead), so a session never silently misses a view."""
        cams = {}
        try:
            for c in self.cfg.cameras:
                try:
                    cams[c.key] = Camera(index=c.index, width=c.width, height=c.height, fps=c.fps)
                except Exception as e:
                    raise RuntimeError(
                        f"camera {c.key!r} (index {c.index}) failed to open: {e}"
                    ) from e
            dead = [key for key, cam in cams.items() if not cam.wait_for_frame()]
            if dead:
                raise RuntimeError(
                    f"camera(s) {dead} delivered no frame; check the device index "
                    "(v4l2-ctl --list-devices) and the connection"
                )
        except Exception:
            for cam in cams.values():
                cam.stop()
            raise
        return cams

    def run(self) -> None:
        """Check the cameras, reset the hands, then collect until quit (blocking)."""
        if not self.cfg.preview:
            for key, cam in self._cameras.items():
                warn_if_dark(f"camera {key!r}", cam.frame)
        elif not preview_cameras(self._cameras.items()):  # before the hands move
            raise SystemExit(0)
        input("[collect] Press Enter to start...")
        for hand in self._hands.values():
            hand.reset(duration=self.cfg.skills.reset_duration)
        self._running = True
        print(self._controls_help())
        threading.Thread(target=self._input_loop, daemon=True).start()
        threading.Thread(target=self._keyboard_loop, daemon=True).start()
        try:
            self._record_loop()
        except KeyboardInterrupt:
            print("\n[collect] interrupted")
        finally:
            self._shutdown()

    def _controls_help(self) -> str:
        grasps = " -> ".join(f"{g} ({GRASP_DESC[g]})" for g in self.hand_model.grasp_cycle)
        left, right = HAND_ACTION_BUTTON["left"], HAND_ACTION_BUTTON["right"]
        lines = [
            f"[collect] controls ({RECORD_BUTTON}, {DISCARD_BUTTON} and the keys act on the "
            "session, the others on the hand of their side):",
            f"  {RECORD_BUTTON} / {RECORD_TOGGLE_KEY}+Enter   start / stop recording (a start "
            "first resets the hands; a stop saves, then resets)",
            f"  {DISCARD_BUTTON} / {DISCARD_KEY}+Enter   discard the current episode",
            f"  {left} / {right}   left / right hand: thumb <-> finger mode",
            "  stick   thumb mode: jog the thumb; finger mode: down closes, up opens the grasp",
            f"  index trigger   finger mode: next grasp ({grasps})",
        ]
        if any(s.rotation_available for s in self._skills.values()):
            lines.append(
                f"  grip   screw mode; then {left} / {right}: start, freeze, back to start"
            )
        lines.append(f"  {QUIT_KEY}+Enter   quit")
        return "\n".join(lines)

    def _input_loop(self) -> None:
        last_warning = 0.0
        while self._running:
            time.sleep(1.0 / self._hz)
            if not self._vr.controller_on:  # no Quest data: hold every command
                now = time.time()
                if now - last_warning > 2.0:
                    print("[collect] Quest controller off (no poses received)")
                    last_warning = now
                continue
            self._input_tick(time.time())

    def _input_tick(self, now: float) -> None:
        buttons = self._vr.buttons
        self._handle_toggle(now)
        self._handle_discard(now)
        self._handle_vr_buttons(buttons, now)
        with self._lock:
            meas = {side: m.copy() for side, m in self._meas.items()}
        commands = {}
        for side, skill in self._skills.items():
            skill.measured = meas[side]
            joystick_x, joystick_y = getattr(buttons, f"{side}_joystick")
            commands[side] = skill.update(
                trigger=getattr(buttons, f"{side}_trigger"),
                joystick_x=joystick_x,
                joystick_y=joystick_y,
                now=now,
            )
        with self._lock:
            self._commands = commands

    def _record_loop(self) -> None:
        period = 1.0 / self._hz
        next_tick = time.time()
        while self._running:
            next_tick += period
            sleep_time = next_tick - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            self._record_tick()

    def _record_tick(self) -> None:
        with self._lock:
            commands = {side: c.copy() for side, c in self._commands.items()}
        meas, ages = {}, {}
        for side in SIDES:
            hand = self._hands.get(side)
            if hand is None:  # a hand that is not connected records its command as its state
                meas[side], ages[side] = commands[side], None
                continue
            hand.send_action(commands[side])
            meas[side] = hand.get_joint_position()
            ages[side] = getattr(hand, "last_readback_age_s", None)
        with self._lock:
            self._meas = {
                side: np.asarray(m, dtype=np.float64).reshape(5, 4).copy()
                for side, m in meas.items()
            }
        poses = self._vr.poses
        images = {key: cam.frame for key, cam in self._cameras.items()}
        self._recorder.step(poses, measured=meas, commands=commands, ages=ages, images=images)
        if self.cfg.preview:  # the window is shown between episodes (cv2 GUI runs on this thread)
            if self._recorder.recording:
                self._close_viewer()
            else:
                self._open_viewer()
                self._update_viewer()

    def _keyboard_loop(self) -> None:
        """Terminal backups for the buttons; only sets flags that the input loop consumes."""
        while self._running:
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            if line == "":  # stdin closed
                break
            key = line.strip().lower()
            if key == RECORD_TOGGLE_KEY:
                self._record_toggle_requested = True
            elif key == DISCARD_KEY:
                self._discard_requested = True
            elif key == QUIT_KEY:
                print("[collect] quitting")
                self._running = False
                break

    # The episode workflow below runs on the input-loop thread.

    def _handle_toggle(self, now: float) -> None:
        """A start resets both hands and begins recording once they have settled (a per-tick
        check, so the reset keeps running); pressing again meanwhile cancels it. A stop saves the
        episode and resets the hands."""
        src = f"{RECORD_BUTTON} / {RECORD_TOGGLE_KEY}"
        if self._record_toggle_requested:
            self._record_toggle_requested = False
            if self._recorder.recording:
                print(f"[collect] recording STOPPED ({src}): saving, then resetting to neutral")
                self._stop_episode(now)
            elif self._pending_start:
                self._pending_start = False
                print(f"[collect] recording start CANCELLED ({src})")
            else:
                print(
                    f"[collect] recording requested ({src}): starts once the hands are back at "
                    "neutral"
                )
                self._pending_start = True
                self._reset_hands(now)
        if self._pending_start and all(s.reset_done for s in self._skills.values()):
            self._pending_start = False
            print(f"[collect] recording STARTED ({src})")
            self._recorder.start()

    def _handle_discard(self, now: float) -> None:
        if not self._discard_requested:
            return
        self._discard_requested = False
        src = f"{DISCARD_BUTTON} / {DISCARD_KEY}"
        if self._recorder.recording:
            print(
                f"[collect] episode DISCARDED ({src}); press {RECORD_BUTTON} to record the next one"
            )
            self._pending_start = False
            self._recorder.discard()
            self._reset_hands(now)
        elif self._pending_start:
            self._pending_start = False
            print(f"[collect] recording start CANCELLED ({src})")
        else:
            print(f"[collect] {src} ignored: not recording")

    def _handle_vr_buttons(self, buttons: ButtonState, now: float) -> None:
        """Rising edges: record / discard become requests for the next input tick; the grip and
        the action button go to that side's skill controller."""
        rec = bool(getattr(buttons, RECORD_BUTTON))
        if rec and not self._record_btn_was_pressed:
            self._record_toggle_requested = True
        self._record_btn_was_pressed = rec
        dis = bool(getattr(buttons, DISCARD_BUTTON))
        if dis and not self._discard_btn_was_pressed:
            self._discard_requested = True
        self._discard_btn_was_pressed = dis

        for side in SIDES:
            skill = self._skills[side]
            if self.hand_model.has_rotation_skill:
                grip = float(getattr(buttons, f"{side}_grip")) > GRIP_THRESHOLD
                if grip and not self._grip_was_pressed[side]:
                    if skill.rotation_available:
                        skill.toggle_screw_mode(now)
                    else:
                        print(
                            f"[collect] {side} grip: screw mode is off (no `rotation_skill:` block)"
                        )
                self._grip_was_pressed[side] = grip
            pressed = bool(getattr(buttons, HAND_ACTION_BUTTON[side]))
            if pressed and not self._action_btn_was_pressed[side]:
                skill.press_action_button(now)
            self._action_btn_was_pressed[side] = pressed

    def _reset_hands(self, now: float) -> None:
        for skill in self._skills.values():
            skill.reset_to_neutral(now)

    def _stop_episode(self, now: float) -> None:
        self._recorder.stop()
        self._reset_hands(now)

    def _open_viewer(self) -> None:
        if self._viewer_open or not self._cameras:
            return
        cv2.namedWindow(VIEWER_WINDOW, cv2.WINDOW_NORMAL)
        cv2.setWindowTitle(VIEWER_WINDOW, f"{VIEWER_WINDOW}: next traj_{self._recorder.next_index}")
        self._viewer_sized = False
        self._viewer_open = True

    def _update_viewer(self) -> None:
        if not self._viewer_open:
            return
        mosaic = camera_mosaic(self._cameras.items())
        if not self._viewer_sized:
            cv2.resizeWindow(VIEWER_WINDOW, int(mosaic.shape[1]), int(mosaic.shape[0]))
            self._viewer_sized = True
        cv2.imshow(VIEWER_WINDOW, mosaic)
        cv2.waitKey(1)

    def _close_viewer(self) -> None:
        if not self._viewer_open:
            return
        cv2.destroyWindow(VIEWER_WINDOW)
        cv2.waitKey(1)
        self._viewer_open = False

    def _shutdown(self) -> None:
        if self._recorder.recording:
            self._stop_episode(time.time())
        self._close_viewer()
        for hand in self._hands.values():
            hand.close()
        for cam in self._cameras.values():
            cam.stop()
        self._vr.stop()
        print("[collect] shut down")
