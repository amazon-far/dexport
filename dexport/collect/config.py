"""Collection config (YAML, schema :class:`CollectConfig`), checked strictly: unknown keys are
errors. ``trajectory.save_dir`` is relative to the working directory; a relative
``rotation_skill.ckpt`` is relative to the repository root. Entry point: :func:`load_config`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from dexport.skills.controller import SkillConfig
from dexport.skills.hand_models import HAND_MODELS, HandModel

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class HandSideConfig:
    enabled: bool = False
    serial: str | None = None  # wuji1: else $WUJI_SERIAL_<SIDE>, else the hand's handedness
    address: str | None = None  # wuji2: else $WUJI_V2_ADDR_<SIDE>, else the robot-LAN default


@dataclass(frozen=True)
class HandsConfig:
    left: HandSideConfig = field(default_factory=HandSideConfig)
    right: HandSideConfig = field(default_factory=HandSideConfig)


@dataclass(frozen=True)
class CameraConfig:
    key: str
    index: int  # OpenCV device index (/dev/video<index>)
    width: int = 224
    height: int = 224
    fps: int = 30


@dataclass(frozen=True)
class TrajectoryConfig:
    save_dir: str = "data"
    tag: str = ""
    save_pose_l: bool = True
    save_quat_l: bool = True
    save_pose_r: bool = False
    save_quat_r: bool = False


@dataclass(frozen=True)
class RotationSkillConfig:
    ckpt: Path  # rotation-skill actor checkpoint


@dataclass(frozen=True)
class CollectConfig:
    """One collection session, as written in YAML:

    hand_model: wuji1 | wuji2              # required
    hand:
      left: {enabled: false}
      right: {enabled: true}               # + an optional selector: serial: <usb-serial>
                                           #   (wuji1), or address: <ip:port> (wuji2, used
                                           #   when discovery finds no hand)
    cameras:                               # one zarr image channel per key
      - {key: right_rgb, index: 0, width: 224, height: 224, fps: 30}
    trajectory: {save_dir: data, tag: "", save_pose_l: true, save_quat_l: true,
                 save_pose_r: false, save_quat_r: false}
    rotation_skill:                        # wuji1 only: enables screw mode
      ckpt: checkpoints/rotation_skill/model.pt
    skills: {...}                          # SkillConfig overrides
    hz: 30                                 # rate of the record and input loops
    preview: true                          # camera window before and between episodes
    """

    hand_model: HandModel
    hand: HandsConfig = field(default_factory=HandsConfig)
    skills: SkillConfig = field(default_factory=SkillConfig)
    cameras: tuple[CameraConfig, ...] = ()
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    rotation_skill: RotationSkillConfig | None = None
    hz: float = 30.0
    preview: bool = True


def _mapping(d, path: str) -> dict:
    if d is None:
        return {}
    if not isinstance(d, dict):
        raise ValueError(f"{path} must be a mapping, got {type(d).__name__}")
    return d


def _check_keys(d: dict, cls, path: str) -> None:
    unknown = sorted(set(d) - {f.name for f in dataclasses.fields(cls)})
    if unknown:
        names = [f"{path}.{k}" if path else str(k) for k in unknown]
        raise ValueError(f"unknown key(s): {', '.join(names)}")


def _bool(v, path: str) -> bool:
    if not isinstance(v, bool):
        raise ValueError(f"{path} must be true or false, got {v!r}")
    return v


def _int(v, path: str) -> int:
    if isinstance(v, bool) or (isinstance(v, float) and not v.is_integer()):
        raise ValueError(f"{path} must be an integer, got {v!r}")
    try:
        return int(v)
    except (TypeError, ValueError):
        raise ValueError(f"{path} must be an integer, got {v!r}") from None


def _float(v, path: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{path} must be a number, got {v!r}")
    return float(v)


def _side(d, path: str, hand_model: str) -> HandSideConfig:
    d = _mapping(d, path)
    _check_keys(d, HandSideConfig, path)
    if hand_model == "wuji1" and "address" in d:
        raise ValueError(f"{path}.address is a wuji2 setting (wuji1 hands are selected by serial)")
    if hand_model == "wuji2" and "serial" in d:
        raise ValueError(f"{path}.serial is a wuji1 setting (wuji2 hands are selected by address)")
    return HandSideConfig(
        enabled=_bool(d.get("enabled", False), f"{path}.enabled"),
        serial=None if d.get("serial") is None else str(d["serial"]),
        address=None if d.get("address") is None else str(d["address"]),
    )


def _cameras(lst) -> tuple[CameraConfig, ...]:
    if lst is None:
        return ()
    if not isinstance(lst, list):
        raise ValueError(f"cameras must be a list, got {type(lst).__name__}")
    cams = []
    for i, c in enumerate(lst):
        path = f"cameras[{i}]"
        c = _mapping(c, path)
        _check_keys(c, CameraConfig, path)
        if "key" not in c or "index" not in c:
            raise ValueError(f"{path} needs a key and an index")
        ints = {k: _int(c[k], f"{path}.{k}") for k in ("index", "width", "height", "fps") if k in c}
        cams.append(CameraConfig(key=str(c["key"]), **ints))
    keys = [c.key for c in cams]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate camera keys: {keys}")
    return tuple(cams)


def config_from_dict(d: dict) -> CollectConfig:
    """Validate a parsed YAML mapping. Raises ValueError for a malformed config and SystemExit for a
    rotation skill requested on a hand that has none."""
    d = _mapping(d, "the config")
    _check_keys(d, CollectConfig, "")
    name = d.get("hand_model")
    if name not in HAND_MODELS:
        raise ValueError(f"hand_model must be one of {sorted(HAND_MODELS)}, got {name!r}")
    model = HAND_MODELS[name]
    # Refused before the block itself is validated: any `rotation_skill:` mapping, even an empty
    # one, asks for the rotation skill.
    if d.get("rotation_skill") is not None and not model.has_rotation_skill:
        raise SystemExit(
            f"{name} has no in-hand rotation skill (the policy is trained for Wuji 1); "
            f"remove the `rotation_skill:` block"
        )

    hand = _mapping(d.get("hand"), "hand")
    _check_keys(hand, HandsConfig, "hand")
    skills = _mapping(d.get("skills"), "skills")
    _check_keys(skills, SkillConfig, "skills")
    traj = _mapping(d.get("trajectory"), "trajectory")
    _check_keys(traj, TrajectoryConfig, "trajectory")

    rotation_skill = None
    if d.get("rotation_skill") is not None:
        s = _mapping(d["rotation_skill"], "rotation_skill")
        _check_keys(s, RotationSkillConfig, "rotation_skill")
        if "ckpt" not in s:
            raise ValueError("rotation_skill.ckpt is required (omit the block to disable it)")
        ckpt = Path(str(s["ckpt"])).expanduser()
        rotation_skill = RotationSkillConfig(ckpt=ckpt if ckpt.is_absolute() else REPO_ROOT / ckpt)

    hz = _float(d.get("hz", 30.0), "hz")
    if hz <= 0:
        raise ValueError(f"hz must be positive, got {hz}")
    return CollectConfig(
        hand_model=model,
        hand=HandsConfig(
            left=_side(hand.get("left"), "hand.left", name),
            right=_side(hand.get("right"), "hand.right", name),
        ),
        skills=SkillConfig(**{k: _float(v, f"skills.{k}") for k, v in skills.items()}),
        cameras=_cameras(d.get("cameras")),
        trajectory=TrajectoryConfig(
            save_dir=str(traj.get("save_dir", "data")),
            tag=str(traj.get("tag") or ""),
            **{
                k: _bool(traj[k], f"trajectory.{k}")
                for k in ("save_pose_l", "save_quat_l", "save_pose_r", "save_quat_r")
                if k in traj
            },
        ),
        rotation_skill=rotation_skill,
        hz=hz,
        preview=_bool(d.get("preview", True), "preview"),
    )


def load_config(path) -> CollectConfig:
    with open(path) as f:
        d = yaml.safe_load(f)
    try:
        return config_from_dict(d)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from None
    except SystemExit as e:
        raise SystemExit(f"{path}: {e}") from None
