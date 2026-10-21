"""Scan recorded zarr stores for hands that were commanded to move but reported no motion.

The recorder prints the same diagnosis when an episode is saved
(:mod:`dexport.collect.channel_health`); this script checks stores already on disk, e.g. before
building a training set. A path is a ``trajectories.zarr`` or a directory holding stores at any
depth. Exits 1 if any store has a warning, 2 if no store is found.

    python scripts/check_frozen_channels.py data/<session>
    python scripts/check_frozen_channels.py data/* --quiet     # only the stores with a warning
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import zarr

from dexport.collect.channel_health import diagnose_hand_channels
from dexport.common.channels import CHANNELS

_HAND_KEYS = tuple(k for ch in CHANNELS.values() for k in (ch.command, ch.measured))


def _is_zarr_group(p: Path) -> bool:
    """A zarr group directory."""
    return p.is_dir() and (p / ".zgroup").exists()


def find_stores(root: Path):
    """Every trajectory store at or under *root*, sorted.

    Two layouts are found: ``<session>/trajectories.zarr/traj_N`` (as recorded) and a zarr group
    with ``traj_N`` directly inside (a flat store).
    """
    if _is_zarr_group(root) and any(root.glob("traj_*")):
        return [root]
    if (root / "trajectories.zarr").exists():
        return [root / "trajectories.zarr"]
    found = sorted(p for p in root.rglob("trajectories.zarr") if p.is_dir())
    if found:
        return found
    # No wrapper anywhere: look for bare session-root groups one level down.
    return (
        sorted(p for p in root.iterdir() if _is_zarr_group(p) and any(p.glob("traj_*")))
        if root.is_dir()
        else []
    )


def scan_store(path: Path, quiet: bool = False, limit: int = 0):
    """Diagnose every trajectory in one store. Returns (n_traj, warn_counter)."""
    g = zarr.open(str(path), mode="r")
    names = sorted(g.group_keys(), key=lambda s: (len(s), s))
    if limit:
        names = names[:limit]

    warns = Counter()
    infos = Counter()
    first_example = {}
    for name in names:
        t = g[name]
        chans = {}
        for k in _HAND_KEYS:
            if k in t:
                chans[k] = np.asarray(t[k][:])
        for f in diagnose_hand_channels(chans):
            key = (f["level"], f["label"], f["meas_key"], f["frozen_joints"])
            if f["level"] == "warn":
                warns[key] += 1
                first_example.setdefault(key, (name, f))
            else:
                infos[key] += 1

    n = len(names)
    if warns:
        print(f"\n{path}  ({n} trajectories)")
        for key, count in warns.most_common():
            name, f = first_example[key]
            pct = 100.0 * count / max(1, n)
            print(
                f"  WARNING: {count}/{n} ({pct:.0f}%)  {f['label']}: "
                f"{f['meas_key']} frozen "
                f"({f['frozen_joints']}/{f['n_joints']} joints), "
                f"{f['cmd_key']} moved {f['cmd_span']:.3f} rad"
            )
            print(f"      e.g. {name}: measured range {f['meas_span']:.2e} rad")
    elif not quiet:
        print(f"\n{path}  ({n} trajectories)")
        print("  ok: every commanded hand reported motion")
        for key, count in infos.most_common():
            print(f"  info: {count}/{n}  {key[1]}: not driven / uncommanded motion")

    return n, warns


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("paths", nargs="+", help="zarr store(s), or directories containing them")
    ap.add_argument("--quiet", action="store_true", help="only print stores that have a warning")
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="check only the first N trajectories per store (0 = all)",
    )
    args = ap.parse_args()

    stores = []
    for p in args.paths:
        found = find_stores(Path(p).expanduser().resolve())
        if not found:
            print(f"[skip] no trajectories.zarr under {p}", file=sys.stderr)
        stores.extend(found)

    if not stores:
        print("nothing to scan", file=sys.stderr)
        return 2

    total_traj = 0
    bad_stores = 0
    total_bad_traj = 0
    for path in stores:
        try:
            n, warns = scan_store(path, quiet=args.quiet, limit=args.limit)
        except Exception as e:  # a half-written store should not abort the sweep
            print(f"\n{path}\n  [error] {type(e).__name__}: {e}", file=sys.stderr)
            continue
        total_traj += n
        if warns:
            bad_stores += 1
            total_bad_traj += max(warns.values())

    print(f"\n{'=' * 60}")
    print(f"scanned {len(stores)} store(s), {total_traj} trajectories")
    if bad_stores:
        print(
            f"WARNING: {bad_stores} store(s) with a dead hand readback "
            f"(~{total_bad_traj} trajectories affected)"
        )
        return 1
    print("ok: no dead hand readbacks found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
