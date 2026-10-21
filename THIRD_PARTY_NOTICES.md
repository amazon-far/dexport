# Third-party notices

DexPort includes code and assets derived from the projects below; their notices are reproduced
here as their licenses require. Dependencies that are installed separately (`pyproject.toml`,
`sim/install.sh`) are not redistributed and keep their own licenses.

## Wuji Technology

The left Wuji hand assets in `sim/assets/wuji_hand_left/` (from
[wuji-description](https://github.com/wuji-technology/wuji-description), `hand/body-with-soft/`)
and the Isaac Lab environment that `sim/` is derived from are distributed by Wuji Technology under
the MIT License:

```
MIT License

Copyright (c) 2025 Wuji Technology

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Learning Dexterous Manipulation Skills from Imperfect Simulations (DexScrew)

`dexport/minbc/` is derived from the minbc code released with *Learning Dexterous Manipulation
Skills from Imperfect Simulations* (DexScrew), which builds on the codebase of HATO (see "Other
attributions"). The screwdriver asset in `sim/assets/screwdriver/` (`screwdriver.urdf`,
`screw_upper_12.stl`) comes from the same release, and the reward and domain-randomization design
of the in-hand rotation environment in `sim/rotation_env/` follows it. MIT License:

```
Copyright (c) 2025 Authors of Learning Dexterous Manipulation Skills from Imperfect Simulations

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## Diffusion Policy

The conditional 1-D U-Net with FiLM conditioning (`dexport/minbc/models/unet.py`) and the
diffusion training and sampling recipe (`dexport/minbc/policy.py`, `dexport/minbc/diffusion.py`)
follow *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*
([real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy)). MIT License:

```
MIT License

Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## UMI: Universal Manipulation Interface

`precise_wait` (`dexport/common/timing.py`) and the relative end-effector pose and 6-D rotation
conventions (`dexport/common/pose.py`) follow
[real-stanford/universal_manipulation_interface](https://github.com/real-stanford/universal_manipulation_interface).
MIT License:

```
MIT License

Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Other attributions

- **HATO** (*Learning Visuotactile Skills with Two Multifingered Hands*): the asynchronous
  inference with per-tick averaging of overlapping action chunks (`dexport/deploy/hato.py`,
  `dexport/deploy/loop.py`) and the joystick mapping of the thumb-control skill
  (`dexport/skills/controller.py`) follow the HATO paper.
- **Separately installed dependencies** (not included here; their own licenses apply): PyRoki
  and jaxls (the arm IK in `dexport/hardware/tianji/pyroki_ik.py` is built from PyRoki's cost
  terms), oculus_reader, Isaac Sim and Isaac Lab, RSL-RL, DINOv2 (code and weights fetched
  through `torch.hub`), wujihandpy and wuji-sdk.
