"""Print the groups and arrays of a trajectory zarr store.

    python scripts/inspect_zarr.py data/<session>/trajectories.zarr [traj_0]

Each array is listed with its shape and dtype, and non-image arrays also with their min / max.
"""

import argparse
import sys

import numpy as np
import zarr


def print_group(group: zarr.Group, indent: int = 0) -> None:
    prefix = "  " * indent
    for name in sorted(group.keys()):
        item = group[name]
        if isinstance(item, zarr.Group):
            print(f"{prefix}{name}/  ({len(list(item.keys()))} items)")
            print_group(item, indent + 1)
            continue
        line = f"{prefix}{name:20s}  shape={str(item.shape):16s}  dtype={item.dtype}"
        if item.ndim < 4 and item.size:  # skip the (T, H, W, 3) image channels
            data = item[:]
            if data.dtype.kind == "f" and np.isnan(data).all():
                line += "  (all NaN)"
            else:
                line += f"  min={np.nanmin(data):+.4f}  max={np.nanmax(data):+.4f}"
        print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("zarr_path", help="path to a trajectories.zarr store")
    ap.add_argument("traj_name", nargs="?", help="only this trajectory, e.g. traj_0")
    args = ap.parse_args()

    store = zarr.open_group(args.zarr_path, mode="r")
    if args.traj_name:
        if args.traj_name not in store:
            sys.exit(f"{args.traj_name!r} not found; available: {sorted(store.keys())}")
        print(f"{args.zarr_path} / {args.traj_name}")
        print_group(store[args.traj_name])
    else:
        print(args.zarr_path)
        print_group(store)


if __name__ == "__main__":
    main()
