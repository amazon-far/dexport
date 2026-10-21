# In-hand rotation skill checkpoint

`model.pt` is a **placeholder**: a randomly initialized actor in the exact RSL-RL checkpoint
format that `sim/train.py` saves and `dexport.skills.rotation.actor.load_rotation_actor` loads.

To use a real skill, train it in simulation (see `sim/README.md`) and replace this file with the
trained checkpoint, e.g.

```bash
cp logs/rotation/<run>/model_<iteration>.pt checkpoints/rotation_skill/model.pt
```

or point `rotation_skill.ckpt` in the collection config at the trained file.
