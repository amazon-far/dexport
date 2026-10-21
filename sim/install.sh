#!/usr/bin/env bash
# Conda env for training the in-hand rotation skill in sim/: Isaac Sim 5.1, Isaac Lab 2.3.2,
# RSL-RL 5.5.1, torch 2.7.0 (CUDA 12.8), numpy 1.26, plus `dexport` (editable, no dependencies).
#
#   bash sim/install.sh [ENV_NAME]   create or update the env (default name: isaaclab)
#   bash sim/install.sh --check      verify the active env (versions, imports, libGLU, EULA)
#
# Installing Isaac Sim accepts the NVIDIA Omniverse EULA
# (https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html):
# the env gets OMNI_KIT_ACCEPT_EULA=YES.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NVIDIA_INDEX="https://pypi.nvidia.com"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
# Verified versions, one group per install step below; --check verifies all of them.
TORCH_PINS=("torch==2.7.0" "torchvision==0.22.0" "torchaudio==2.7.0")
ISAACSIM_PIN="isaacsim==5.1.0.0"
ISAACLAB_PIN="isaaclab==2.3.2.post1"
# tensordict must be pinned with rsl-rl-lib: a newer tensordict pulls a newer torch, which breaks
# Isaac Sim.
RL_PINS=("rsl-rl-lib==5.5.1" "tensordict==0.9.1")
# Installed last, over the newer versions the steps before pull in. typeguard>=4.5 needs
# typing_extensions>=4.13, which conflicts with the 4.12.2 pin.
REPINS=(
  "numpy==1.26.0" "click==8.1.7" "typing-extensions==4.12.2" "idna==3.10" "packaging==23.0"
  "typeguard<4.5"
)
ISAACLAB_RL_PIN="isaaclab_rl==0.4.7"  # Isaac Lab's RSL-RL wrapper, built from the isaaclab wheel
ALL_PINS=(
  "${TORCH_PINS[@]}" "$ISAACSIM_PIN" "$ISAACLAB_PIN" "${RL_PINS[@]}" "${REPINS[@]}"
  "$ISAACLAB_RL_PIN"
)
EXTRAS=(tensorboard imageio imageio-ffmpeg)  # unpinned
OPTIONAL_EXTRAS=(wandb)                      # unpinned, for train.py --wandb only

check_env() {
  python - "$REPO" "${OMNI_KIT_ACCEPT_EULA:-}" "${TORCH_INDEX##*/}" "${ALL_PINS[*]}" \
    "${EXTRAS[*]}" "${OPTIONAL_EXTRAS[*]}" <<'PY'
import importlib.metadata as md
import re
import subprocess
import sys

repo, eula, torch_build, pins, extras, optional_extras = sys.argv[1:]
problems, notes = [], []
if sys.version_info[:2] != (3, 11):
    problems.append(f"python {sys.version.split()[0]} (want 3.11)")
try:
    from packaging.specifiers import SpecifierSet
except ImportError:
    SpecifierSet = None
    problems.append("packaging not installed")
for pin in pins.split():
    name, spec = re.fullmatch(r"([A-Za-z0-9_.-]+)([=<].+)", pin).groups()
    try:
        got = md.version(name)
    except md.PackageNotFoundError:
        problems.append(f"{name} not installed (want {spec})")
        continue
    if SpecifierSet is not None and got not in SpecifierSet(spec):
        problems.append(f"{name} {got} (want {spec})")
    if name == "torch" and not got.endswith("+" + torch_build):
        problems.append(f"torch {got} is not the {torch_build} build")
for names, missing in ((extras, problems), (optional_extras, notes)):
    for name in names.split():
        try:
            md.version(name)
        except md.PackageNotFoundError:
            missing.append(f"{name} not installed")
# -P: as for `python sim/train.py`, the current directory must not provide `dexport`.
if sys.version_info >= (3, 11):
    probe = subprocess.run([sys.executable, "-P", "-c", "import dexport.skills.rotation.contract"],
                           capture_output=True, text=True)
    if probe.returncode != 0:
        problems.append(f"cannot import dexport (pip install --no-deps -e {repo})")
if eula.upper() != "YES":
    problems.append("OMNI_KIT_ACCEPT_EULA is not YES")
if "libGLU.so.1" not in subprocess.run(["ldconfig", "-p"], capture_output=True, text=True).stdout:
    problems.append("libGLU.so.1 missing (sudo apt-get install -y libglu1-mesa)")
for p in notes:
    print(f"[install.sh --check] note: {p} (needed only for train.py --wandb)")
for p in problems:
    print(f"[install.sh --check] {p}")
print("[install.sh --check] " + ("FAILED" if problems else "ok"))
sys.exit(1 if problems else 0)
PY
}

if [ "${1:-}" = "--check" ]; then
  check_env
  exit $?
fi

ENV_NAME="${1:-isaaclab}"
if ! grep -q libGLU.so.1 <<<"$(ldconfig -p)"; then
  echo "Isaac Sim needs libGLU: sudo apt-get install -y libglu1-mesa" >&2
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
grep -qE "^${ENV_NAME}\s" <<<"$(conda env list)" || conda create -n "$ENV_NAME" python=3.11 -y
conda activate "$ENV_NAME"
export OMNI_KIT_ACCEPT_EULA=YES

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT
# Isaac Sim requires torch==2.7.0, and PyPI's torch 2.7.0 is another CUDA build: install the CUDA
# 12.8 build first and constrain every later step to it.
printf '%s\n' "${TORCH_PINS[@]}" > "$workdir/torch.txt"
pip install -U "${TORCH_PINS[@]}" --index-url "$TORCH_INDEX"
pip install "isaacsim[all,extscache]${ISAACSIM_PIN#isaacsim}" -c "$workdir/torch.txt" \
  --extra-index-url "$NVIDIA_INDEX"
pip install "$ISAACLAB_PIN" -c "$workdir/torch.txt" --extra-index-url "$NVIDIA_INDEX"
pip install "${RL_PINS[@]}" "${EXTRAS[@]}" "${OPTIONAL_EXTRAS[@]}" -c "$workdir/torch.txt"
pip install "${REPINS[@]}"
# The wrapper's source ships inside the installed isaaclab package; building a copy keeps the build
# files out of site-packages.
ISAACLAB_DIR="$(python -P -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("isaaclab").origin))')"
cp -r "$ISAACLAB_DIR/source/isaaclab_rl" "$workdir/isaaclab_rl"
pip install --no-deps "$workdir/isaaclab_rl"
pip install --no-deps -e "$REPO"
conda env config vars set -n "$ENV_NAME" OMNI_KIT_ACCEPT_EULA=YES

check_env
echo "Done. Activate with: conda activate $ENV_NAME"
