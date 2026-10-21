"""Policy and training configuration (defaults: the training recipe) and the saved-config loader.

Observation keys: ``img`` (camera images in ``data.im_key`` order), ``ee_pos`` / ``ee_quat``
(end-effector pose relative to the current frame; ``ee_quat`` holds the 6-D rotation),
``hand_qpos`` (measured hand joints) and, for bimanual policies, the right-side ``*_r`` keys plus
``rel_ee`` (right pose in the left frame). The action is one 9-D relative pose per arm (pos +
rot6d, arms first) followed by one 20-D absolute target per hand: 29-D, or 58-D bimanual.
Every run writes its ``MinBCConfig`` to ``config.json``; ``config_from_dict`` reads it back. The
field docstrings are the ``--help`` text of ``python -m dexport.minbc.train``.
"""

import copy
import dataclasses
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

#: Observation keys in the order the policy concatenates their features.
OBS_KEY_ORDER = (
    "img",
    "ee_pos",
    "ee_quat",
    "hand_qpos",
    "ee_pos_r",
    "ee_quat_r",
    "hand_qpos_r",
    "rel_ee",
)
#: Width of each proprio key in the zarr / dataset buffers (ee_quat is an xyzw quaternion).
RAW_OBS_DIMS = {
    "ee_pos": 3,
    "ee_quat": 4,
    "hand_qpos": 20,
    "ee_pos_r": 3,
    "ee_quat_r": 4,
    "hand_qpos_r": 20,
    "rel_ee": 9,
}
#: Width of each proprio key as fed to its encoder: the ``ee_quat*`` observations are the 6-D
#: rotation relative to the current frame (the key name is kept for checkpoint compatibility).
POLICY_OBS_DIMS = {**RAW_OBS_DIMS, "ee_quat": 6, "ee_quat_r": 6}
#: Keys only bimanual policies observe: the right arm / hand, and the right pose in the left frame.
BIMANUAL_ONLY_KEYS = ("hand_qpos_r", "ee_pos_r", "ee_quat_r", "rel_ee")
#: data_key of the bimanual policies.
BIMANUAL_DATA_KEY = ("img", "hand_qpos", "ee_pos", "ee_quat") + BIMANUAL_ONLY_KEYS
#: Action width per arm (relative pose: 3-D position + 6-D rotation) and per hand (joint targets).
ARM_ACTION_DIM = 9
HAND_ACTION_DIM = 20
#: Key of an inference window step's stacked camera images (n_cam, H, W, 3) uint8, in ``im_key``
#: order (``dexport.minbc.inference``); not a camera name.
WINDOW_IMAGES_KEY = "base_rgb"
#: Default thresholds of the Quest tracking-jump detector (``data/jolt_split.py``).
JOLT_Z_THRESH = 4.0
JOLT_FLOOR_M = 0.015
JOLT_ABS_GATE_M = 0.03


@dataclass(frozen=True)
class EncoderConfig:
    """MLP widths of the proprio encoders, one field per observation key."""

    ee_pos_net_dim: Tuple[int, ...] = (256, 256, 64)
    """Layer widths of the ee_pos encoder."""
    ee_quat_net_dim: Tuple[int, ...] = (256, 256, 64)
    """Layer widths of the ee_quat encoder."""
    hand_qpos_net_dim: Tuple[int, ...] = (256, 256, 64)
    """Layer widths of the hand_qpos encoder."""
    ee_pos_r_net_dim: Tuple[int, ...] = (256, 256, 64)
    """Layer widths of the ee_pos_r encoder."""
    ee_quat_r_net_dim: Tuple[int, ...] = (256, 256, 64)
    """Layer widths of the ee_quat_r encoder."""
    hand_qpos_r_net_dim: Tuple[int, ...] = (256, 256, 64)
    """Layer widths of the hand_qpos_r encoder."""
    rel_ee_net_dim: Tuple[int, ...] = (256, 256, 64)
    """Layer widths of the rel_ee encoder."""


@dataclass(frozen=True)
class DPConfig:
    """Diffusion policy: observation window, action chunk, noise schedule and U-Net size."""

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    obs_horizon: int = 2
    """Observation frames per sample."""
    act_horizon: int = 8
    """Actions per chunk that a window must cover before it may edge-pad past an episode end."""
    total_horizon: int = 16
    """Actions per predicted chunk."""
    obs_current_step: bool = True
    """True: the observed frames end at the current frame f_c (the frame the actions and EE
    observations are relative to). False: the frames strictly before f_c are observed."""
    diffusion_iters: int = 50
    """Diffusion steps of the training noise schedule (DDIM, squared-cosine betas, epsilon
    prediction); inference samples with the same number unless told otherwise."""
    diffusion_method: Literal["ddim", "ddpm"] = "ddim"
    """Scheduler family of the noise schedule."""
    eval_diffusion_iters: int = 10
    """Sampling steps of the test evaluation that selects model_best.ckpt (0: diffusion_iters)."""
    down_dims: Tuple[int, ...] = (512, 1024, 2048)
    """U-Net channels per level."""


@dataclass(frozen=True)
class OptimConfig:
    """Optimization: AdamW, cosine learning-rate schedule with 500 warmup steps, EMA weights."""

    batch_size: int = 64
    """Training batch size."""
    num_train_steps: int = 500_000
    """Gradient steps."""
    eval_freq: int = 10_000
    """Steps between test-split evaluations (the best one is saved as model_best.ckpt)."""
    save_freq: int = 10_000
    """Steps between model_last.ckpt saves."""
    weight_decay: float = 0.01
    """AdamW weight decay."""
    learning_rate: float = 1e-4
    """Learning rate (the DINOv2 backbone trains at 0.1x)."""
    num_workers: int = 8
    """DataLoader worker processes (streamed images are decoded in the workers: use 16+)."""


@dataclass(frozen=True)
class DataConfig:
    """Training data: observation keys, cameras, action labels, preprocessing, image augmentation
    (the im_aug / im_grayscale / im_blur fields are the training augmentations)."""

    data_key: Tuple[str, ...] = ("img", "hand_qpos", "ee_pos", "ee_quat")
    """Observation keys: img, ee_pos, ee_quat, hand_qpos, and for bimanual policies ee_pos_r,
    ee_quat_r, hand_qpos_r, rel_ee. ee_pos and ee_quat are required."""
    im_key: Tuple[str, ...] = ("left_rgb",)
    """Cameras, in the order the policy sees them: left_rgb, right_rgb (base_rgb: see
    zarr_io.CAMERA_ALIASES)."""
    dinov2_model: str = "dinov2_vits14"
    """torch.hub name of the image backbone (fine-tuned; one network shared by all cameras)."""
    im_height: int = 224
    """Height (pixels) of the images the backbone sees."""
    im_width: int = 224
    """Width (pixels) of the images the backbone sees."""
    im_stream: bool = False
    """Read image frames from the zarr store per sample instead of holding them all in RAM."""
    bimanual: bool = False
    """Two arms and two hands (58-D actions); data_key then needs the right-side keys."""
    hand_side: Literal["left", "right"] = "left"
    """Single-arm data: which arm/hand's channels to read."""
    hand_action_source: Literal["command", "state"] = "command"
    """Hand action label: the commanded targets, or the measured joints (state-as-action)."""
    jolt_split: bool = True
    """Split episodes at Quest tracking jumps so no window spans one (data/jolt_split.py)."""
    jolt_z_thresh: float = JOLT_Z_THRESH
    """Tracking-jump test: a step at least this many times the trajectory's 95th-percentile step."""
    jolt_floor_m: float = JOLT_FLOOR_M
    """Tracking-jump test: and at least this long (m)."""
    jolt_abs_gate_m: float = JOLT_ABS_GATE_M
    """Tracking-jump test: any step at least this long (m) is a jump."""
    norm_percentile: float = 2.0
    """Observation and arm-action bounds are the [p, 100 - p] percentiles (0: min/max)."""
    im_aug_crop_ratio: float = 0.95
    """Random crop of this fraction of the image side, resized back (evaluation: center crop)."""
    im_aug_jitter: Tuple[float, float, float, float] = (0.3, 0.4, 0.5, 0.08)
    """Color jitter: brightness, contrast, saturation, hue."""
    im_grayscale_p: float = 0.1
    """Probability of a grayscale conversion."""
    im_blur_p: float = 0.2
    """Probability of a Gaussian blur."""
    im_blur_kernel: int = 5
    """Gaussian blur kernel size (an even size rounds up to the next odd one; at least 3)."""
    im_blur_sigma: Tuple[float, float] = (0.1, 1.5)
    """Range of the Gaussian blur sigma."""

    def __post_init__(self):
        unknown = [k for k in self.data_key if k not in OBS_KEY_ORDER]
        if unknown:
            raise ValueError(f"data_key {unknown} not supported (supported: {OBS_KEY_ORDER})")
        if not {"ee_pos", "ee_quat"} <= set(self.data_key):
            raise ValueError(
                "data_key needs ee_pos and ee_quat: the arm actions are relative poses"
            )
        right = [k for k in self.data_key if k in BIMANUAL_ONLY_KEYS]
        if self.bimanual and not {"ee_pos_r", "ee_quat_r"} <= set(self.data_key):
            raise ValueError("a bimanual data_key needs ee_pos_r and ee_quat_r")
        if right and not self.bimanual:
            raise ValueError(f"data_key {right} needs data.bimanual")
        if "img" in self.data_key and not self.im_key:
            raise ValueError("data_key has img but im_key names no camera")

    @property
    def n_arms(self) -> int:
        return 2 if self.bimanual else 1

    @property
    def arm_action_dim(self) -> int:
        return ARM_ACTION_DIM * self.n_arms

    @property
    def hand_action_dim(self) -> int:
        return HAND_ACTION_DIM * self.n_arms

    @property
    def action_dim(self) -> int:
        return self.arm_action_dim + self.hand_action_dim


@dataclass(frozen=True)
class MinBCConfig:
    """One training run: data sources, run directory, and the policy / optimizer / data settings."""

    seed: int = 0
    """Seed of the network initialization, data order and augmentation."""
    gpu: str = "0"
    """CUDA device index, or "cpu"."""
    data_dir: str = ""
    """Directory that --train-data and --test-data are relative to."""
    output_dir: str = "outputs"
    """Parent of the run directory."""
    output_name: str = ""
    """Name of the run directory <output_dir>/<output_name>."""
    train_data: str = ""
    """Train stores joined by "+", one per co-training source, e.g.
    "lab/train/trajectories.zarr+wild/train/trajectories.zarr"."""
    test_data: str = ""
    """Test store; model_best.ckpt has the lowest action error on it."""
    train_sample_mode: Literal["concat", "balanced", "weighted"] = "concat"
    """How batches draw from the train sources: concat (uniformly over all windows), weighted
    (the shares in train_source_mix) or balanced (every source the same share)."""
    train_source_mix: str = ""
    """Shares of the weighted mode, e.g. "0:0.5,1:0.5" (see data/sampling.py)."""
    dp: DPConfig = field(default_factory=DPConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    data: DataConfig = field(default_factory=DataConfig)
    use_wandb: bool = False
    """Also log to Weights & Biases (needs the wandb package)."""
    wandb_project: str = "minbc"
    """Weights & Biases project."""
    wandb_entity: Optional[str] = None
    """Weights & Biases entity (None: the account's default)."""


# Loading a saved config.json.
#: Values assumed for keys a saved config.json may lack (they differ from the dataclass defaults).
_ABSENT_KEY_DEFAULTS = {
    ("dp", "obs_current_step"): False,
    ("dp", "down_dims"): (256, 512, 1024),
    ("data", "im_key"): ("base_rgb",),
}
#: Saved values this code can load: anything else would change the network or its inputs.
_SUPPORTED_VALUES = {
    ("policy_type",): ("dp",),
    ("data", "use_relative_arm_action"): (True,),
    ("data", "pred_hand_act"): (True,),
    ("data", "pred_head_act"): (False,),
    ("data", "pred_waist_act"): (False,),
    ("data", "relative_hand_action"): (False,),
    ("data", "dexumi_obs"): (False,),
    ("data", "im_channel"): (3,),
    ("dp", "encoder", "im_net_output_dim"): (None,),
}
_MISSING = object()


def _lookup(d, path):
    for key in path:
        if not isinstance(d, dict) or key not in d:
            return _MISSING
        d = d[key]
    return d


def _unsupported(path, value, allowed):
    return ValueError(
        f"config {'.'.join(path)}={value!r} is not supported: this code implements only "
        f"{' / '.join(repr(a) for a in allowed)}."
    )


def _check_supported(raw: dict) -> None:
    for path, allowed in _SUPPORTED_VALUES.items():
        value = _lookup(raw, path)
        if value is not _MISSING and value not in allowed:
            raise _unsupported(path, value, allowed)
    data = raw.get("data", {})
    encoder = _lookup(data, ("im_encoder",))
    if encoder is not _MISSING and "img" in data.get("data_key", ()) and encoder != "DINO":
        raise _unsupported(("data", "im_encoder"), encoder, ("DINO",))
    n_arms = 2 if data.get("bimanual", False) else 1
    action_dim = (ARM_ACTION_DIM + HAND_ACTION_DIM) * n_arms
    base_action_dim = _lookup(data, ("base_action_dim",))
    if base_action_dim is not _MISSING and base_action_dim != action_dim:
        raise _unsupported(("data", "base_action_dim"), base_action_dim, (action_dim,))
    # ee_quat*_input_dim is not checked: saved configs record the raw quaternion width (4) while
    # the encoder takes the 6-D rotation.
    for key in ("ee_pos", "hand_qpos", "ee_pos_r", "hand_qpos_r", "rel_ee"):
        path = ("dp", "encoder", f"{key}_input_dim")
        value = _lookup(raw, path)
        if value is not _MISSING and value != POLICY_OBS_DIMS[key]:
            raise _unsupported(path, value, (POLICY_OBS_DIMS[key],))


def _tuplify(value):
    if isinstance(value, list):
        return tuple(_tuplify(v) for v in value)
    return value


def _merge(obj, d: dict):
    changes = {}
    for f in dataclasses.fields(obj):
        if f.name in d:
            current = getattr(obj, f.name)
            if dataclasses.is_dataclass(current):
                changes[f.name] = _merge(current, d[f.name])
            else:
                changes[f.name] = _tuplify(d[f.name])
    return dataclasses.replace(obj, **changes)


def config_from_dict(raw: dict) -> MinBCConfig:
    """Build a config from a saved ``config.json`` dict.

    Missing keys take their defaults (or the value in ``_ABSENT_KEY_DEFAULTS``), keys this code
    does not know are ignored, and JSON lists become tuples. ``data.data_key`` is required. A
    saved value that selects an unsupported policy, encoder or action/observation layout raises
    ValueError.
    """
    raw = copy.deepcopy(raw)
    if "data_key" not in raw.get("data", {}):
        raise ValueError("config.json has no data.data_key")
    for path, value in _ABSENT_KEY_DEFAULTS.items():
        node = raw
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node.setdefault(path[-1], value)
    _check_supported(raw)
    return _merge(MinBCConfig(), raw)
