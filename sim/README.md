# In-hand rotation skill (Isaac Lab)

PPO training of the in-hand rotation skill that the Wuji 1 collector runs during screwdriver
demonstrations: a fixed left Wuji hand turns a screwdriver handle with its thumb, index and middle
fingers. `sim/` is its own environment (Isaac Sim / Isaac Lab); it
uses only the numpy modules `dexport.common.joints` and `dexport.skills.rotation.contract`, which
it shares with the robot-side runner.

## Install

```bash
bash sim/install.sh            # conda env "isaaclab" (or: bash sim/install.sh ENV_NAME)
conda activate isaaclab
bash sim/install.sh --check    # verify the active env
```

Needs Linux, an NVIDIA GPU and `libglu1-mesa`. The script pins Isaac Sim 5.1.0, Isaac Lab 2.3.2,
RSL-RL 5.5.1 (with tensordict 0.9.1), torch 2.7.0 (the CUDA 12.8 build, installed first from the
PyTorch index) and numpy 1.26.0, installs Isaac Lab's RSL-RL wrapper from the source shipped in the
Isaac Lab wheel, installs this repo with `pip install --no-deps -e .`, and sets
`OMNI_KIT_ACCEPT_EULA=YES` in the env (installing Isaac Sim accepts the NVIDIA Omniverse EULA).

## Train

```bash
python sim/train.py --headless
```

## Evaluate

```bash
python sim/eval.py --headless --checkpoint logs/rotation/<run>/model_1199.pt [--nominal] [--video]
```

## Use a trained actor on the robot

```bash
cp logs/rotation/<run>/model_1199.pt checkpoints/rotation_skill/model.pt
```

or point `rotation_skill.ckpt` in the collection config at the file. The robot loads only the
actor, with torch (`dexport.skills.rotation.actor`), and `dexport.skills.rotation.runner` rebuilds
the observation and action of this environment from the hand encoders.

## Layout

```
sim/
├── train.py, eval.py           entry points
├── install.sh                  conda env
├── rotation_env/
│   ├── env_cfg.py              scene, observations, actions, events, rewards, terminations
│   ├── env.py                  build_env_cfg(), make_env()
│   ├── ppo.py                  PPO settings, actor loader
│   └── mdp/                    action, observation, reward, termination and event terms
└── assets/                     hand and screwdriver (see assets/README.md)
```
