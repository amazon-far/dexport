# DexPort: detailed usage

## Repository layout

```text
.
├── dexport/              Python package; everything that runs without Isaac Sim
│   ├── common/           joint orders, pose math, channel names, terminal keys, timing
│   ├── hardware/         camera, Quest reader, Wuji 1 / Wuji 2 drivers, Tianji arms with PyRoki IK
│   ├── skills/           the Quest-input skill controller, hand models, in-hand rotation runtime
│   ├── collect/          demonstration collector, collection config, zarr recorder
│   ├── minbc/            diffusion policy: config and presets, data, models, training, inference
│   ├── calib/            capture and solve of the controller-to-flange mount
│   └── deploy/           policy and replay deployment through the HATO control loop
├── sim/                  Isaac Lab PPO training of the in-hand rotation skill (its own environment)
│   ├── rotation_env/     environment configuration and MDP terms
│   └── assets/           Wuji hand and screwdriver assets
├── configs/collect/      one collection config per task
├── checkpoints/          the placeholder in-hand rotation actor
├── hardware/             3D-printed parts (STL) and bill of materials
└── scripts/              dataset tools (inspect, check, split)
```

## Install

Python 3.11 on Linux. Clone the repository and install it in editable mode from its root (a
relative `rotation_skill.ckpt` in a collection config resolves against the checkout), with the
extra of each machine's role:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[collect]"   # or train, deploy, calib, wandb
```

### Collection laptop

```bash
sudo apt install adb git-lfs && git lfs install   # before pip: the Quest reader's app is in Git LFS
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: CPU torch is enough
pip install -e ".[collect]"
```

- **Quest 2**: developer mode, connected by USB (`adb devices` lists it); the reader installs its
  app on the headset on first connect.
- **Wuji 1** (USB): each hand is opened by its USB serial (`serial:` in the config, or
  `WUJI_SERIAL_LEFT` / `WUJI_SERIAL_RIGHT`), else by the handedness flashed in the hand.
- **Wuji 2** (Ethernet): found by discovery, else at `address:` / `WUJI_V2_ADDR_<SIDE>`, else at
  the robot-LAN defaults 192.168.1.110:7447 (left) and 192.168.1.111:7447 (right). Both hands ship
  as .111, so re-IP the left one before using two.
- The Wuji SDKs keep logs and caches under `~/.wuji/`, and wuji-sdk uploads anonymous usage data
  unless `WUJI_TELEMETRY_DISABLE=1` is set.
- **Cameras**: `python -m dexport.hardware.camera --index 0 --show` checks one device
  (index N is `/dev/videoN`).

### GPU training machine

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # see below
pip install -e ".[train]"      # ".[train,wandb]" for --use-wandb
```

The first line installs torch for CUDA 12.6 (any index whose CUDA version the NVIDIA driver
supports works). Without it, pip installs PyPI's default torch build, for torch 2.14 a CUDA 13
build that needs driver 580 or newer; on an older driver the policy then fails to load on the GPU.

The DINOv2 ViT-S/14 encoder comes through `torch.hub` (`facebookresearch/dinov2`): the first use
needs network access or a warm `$TORCH_HOME/hub` cache. Loading a trained checkpoint for
deployment needs the hub code too.

### Robot PC

```bash
sudo apt install build-essential   # pyroki's dependency pyliblzfse is compiled on Linux
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # as for training
pip install -e ".[deploy]"         # mount calibration: git-lfs as above, then ".[calib]"
```

- `tj_marvin` (the Marvin SDK): installed in the environment or on `PYTHONPATH`. Its YAML comes
  from `--marvin-config` or `$TJ_MARVIN_ROOT/configs/marvin_config.yml`.
- The two-arm Marvin CCS-M6 URDF: `--urdf` or `TIANJI_URDF`.
- The gravity-compensation tool folder of each arm with its Wuji hand, written by tj_marvin's
  dynamics identification: `TianjiConfig.tool_payload` in `dexport/hardware/tianji/arms.py`
  (default `hw_test/dyn_id/output/left_arm_A_hand_wuji` and `right_arm_B_hand_wuji`, relative to
  the marvin YAML).
- A CUDA GPU for the policy: deployment refuses to start when inference is too slow (twice the
  latency must fit in the 8 kept steps of a chunk).
- jaxls is installed from its main branch (the URL pyroki requires); it was tested at commit
  `50a58be`, which can be pinned if a later version breaks pyroki.

### Simulation machine

`bash sim/install.sh` creates the `isaaclab` conda env (Isaac Sim 5.1, Isaac Lab 2.3.2,
RSL-RL 5.5.1) and installs this repository into it without dependencies; see `sim/README.md`.

### Environment variables

| variable | used by | meaning |
|---|---|---|
| `TIANJI_URDF` | deploy, calibration | the two-arm URDF (or `--urdf`) |
| `TJ_MARVIN_ROOT` | deploy, calibration | tj_marvin checkout holding `configs/marvin_config.yml` (or `--marvin-config`) |
| `WUJI_SERIAL_LEFT`, `WUJI_SERIAL_RIGHT` | Wuji 1 | USB serial of that side's hand |
| `WUJI_V2_ADDR_LEFT`, `WUJI_V2_ADDR_RIGHT` | Wuji 2 | `ip:port` used when discovery finds no hand |
| `WUJI_V2_READBACK_PROBE_S` | Wuji 2 | connect-time readback probe in s (0 skips it on a bench that only sends commands) |
| `WUJI_TELEMETRY_DISABLE` | wuji-sdk (Wuji 2) | `1` turns off the SDK's anonymous usage-data upload |
| `WUJI_CAMERA_FOURCC` | cameras | capture format (`MJPG`, `YUYV`; unset: OpenCV's negotiation); keep it the same for collection and deployment |
| `TORCH_HOME` | training, deployment | torch.hub cache with DINOv2 |
| `JAX_PLATFORMS` | deploy, calibration | the IK runs on the CPU unless this says otherwise |
| `WANDB_MODE`, `WANDB_USERNAME` | `--use-wandb`, `sim/train.py --wandb` | e.g. `WANDB_MODE=offline`; the W&B entity of the sim runs |

## Usage

### 1. Collect demonstrations

```bash
python -m dexport.collect --config configs/collect/wuji1_pnp.yaml
```

| config | task | hand | cameras |
|---|---|---|---|
| `wuji1_pnp.yaml` | vegetable pick-and-place | Wuji 1, right | `right_rgb` (wrist) |
| `wuji1_screwdriver.yaml` | screwdriver, with the in-hand rotation skill | Wuji 1, left | `left_rgb` (wrist) |
| `wuji2_drawer.yaml` | drawer stowing | Wuji 2, right | `right_rgb` (wrist), `left_rgb` (third person) |
| `wuji2_scan.yaml` | bimanual barcode scanning | two Wuji 2 | `left_rgb`, `right_rgb` (wrists) |

All config keys and their defaults are documented in `dexport/collect/config.py`. The collector
first shows the cameras (any key continues, `q` quits) and waits for Enter. The controls (B, Y and
the keys act on the session, the others on the hand of their side):

| input | action |
|---|---|
| B (or `x` + Enter) | start / stop recording: a start first resets the hands, a stop saves the episode and then resets them |
| Y (or `d` + Enter) | discard the current episode |
| X (left) / A (right) | switch between thumb and finger mode; in screw mode: start the rotation policy, freeze it, ease back to its start grasp |
| grip | enter / leave screw mode (Wuji 1, with a `rotation_skill:` block in the config) |
| stick | thumb mode: x flexes, y abducts the thumb; finger mode: down closes, up opens the selected grasp |
| index trigger | finger mode: next grasp (Wuji 1: power, pinch; Wuji 2: power, two-finger) |
| `q` + Enter | quit |

Each session is one zarr store (`T` rows per episode, one per 30 Hz tick; the channel names are
defined in `dexport/common/channels.py`):

```text
data/<MMDD_HHMM>_<tag>/trajectories.zarr/traj_<i>/
    timestamp                             (T,)
    action, action_r                      (T, 5, 4)  commanded hand targets, left / right hand
    hand_joints, hand_joints_r            (T, 5, 4)  measured hand joints
    pose_l, quat_l, pose_r, quat_r        (T, 3), (T, 4)  Quest controller poses (xyzw), as configured
    <camera key>, e.g. right_rgb          (T, 224, 224, 3) uint8 RGB
    hand_joints_age, hand_joints_r_age    (T,)  age of the hand readback in s (Wuji 2)
```

Check a session:

```bash
python scripts/inspect_zarr.py data/<session>/trajectories.zarr traj_0
# exits 1 if a hand's measured joints never moved while it was commanded (a dead readback)
python scripts/check_frozen_channels.py data/<session>
```

### 2. Prepare the data

Pool sessions into a train/test split of symlinks (`--dry-run` prints it first):

```bash
python scripts/make_split.py --sessions data/*_pnp --out data/splits/pnp --test-count 10
```

This writes `data/splits/pnp/{train,test}/trajectories.zarr` and `MANIFEST.json`.

### 3. Train a policy

Example command:

```bash
python -m dexport.minbc.train pnp --data-dir data/splits \
    --train-data pnp/train/trajectories.zarr --test-data pnp/test/trajectories.zarr
```

### 4. Calibrate the controller-to-flange mount

The policy's arm actions are Quest-controller poses; deployment turns them into flange targets
through the mount, the fixed transform between the controller and the flange. The calibration
fits it from paired poses: each sample is the flange pose and the controller pose at the same
moment.

Bolt the hand-camera module, with the controller bracket, to the flange of arm A (left) or B
(right), and keep the Quest headset still with the controller in view. The bracket puts the
controller between flange and hand (flange → bracket → hand instead of flange → hand), which adds
12 mm of standoff, the spacer. The fit accounts for it; deployment assumes the bracket is off
unless `--spacer` is given.

```bash
python -m dexport.calib.capture --mode probe --arm A                         # tracking check, Ctrl-C stops
python -m dexport.calib.capture --mode jog --arm A --out data/calib/arm_a.npz
python -m dexport.calib.solve --npz data/calib/arm_a.npz                     # --hand wuji2 on a Wuji 2 rig
```

In the jog, the keys move the flange and Space records a pair once the arm has settled (`?` lists
the keys, `x` saves and quits). Record about 30 poses that tilt the flange by at least 30° about
several axes, then paste the `QUEST_TO_*` lines the solver prints into `dexport/deploy/config.py`.

### 5. Deploy

Replay a recording first, then run the policy:

```bash
# a Wuji 2 session (drawer, scanning) also needs --hand wuji2
python -m dexport.deploy --replay data/<session>/trajectories.zarr --traj traj_0 --arm B
python -m dexport.deploy --ckpt outputs/pnp
```

- The arms come from the checkpoint: a left-hand policy drives arm A, a right-hand one arm B, a
  bimanual one both. `--ckpt RUN` loads `model_last.ckpt`; `--ckpt RUN/model_best.ckpt` loads the
  checkpoint with the lowest test error.
- Keys in the terminal: a key press moves the arms home (`q` quits); during an episode Space
  pauses and resumes and Esc (or an arrow key) ends it; after an episode `r` runs another one.

### 6. The in-hand rotation skill

Training and evaluation are in `sim/README.md`. To use a trained actor, replace the
placeholder:

```bash
cp logs/rotation/<run>/model_1199.pt checkpoints/rotation_skill/model.pt
```

or point `rotation_skill.ckpt` in the collection config at the file.
