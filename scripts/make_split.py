"""Pool recording sessions into a train/test split of trajectory symlinks.

    python scripts/make_split.py --sessions data/*_scan --out data/splits/scan --test-count 6

Sessions (directories or their trajectory stores) are ordered by name and their trajectories by
index, renumbered ``traj_0..`` across the pool; ``--test-count`` evenly spaced trajectories (or
``--test-trajs session/traj_K,...``) form the test split. Writes ``<out>/{train,test}/
trajectories.zarr`` (absolute symlinks; pass them as ``--train-data`` / ``--test-data``) and
``<out>/MANIFEST.json``. Only the new ``--out`` is written (``--dry-run``: nothing); sessions are
only read.
"""

import argparse
import json
import os

import natsort
import numpy as np
import zarr

from dexport.minbc.data.zarr_io import session_name, session_store, trajectory_names


def pooled_trajectories(paths):
    """[(session, trajectory, realpath)] in session-name order, trajectories in index order."""
    sessions = []
    for path in paths:
        store = session_store(path)
        if store is None:
            raise SystemExit(f"{path} is neither a session directory nor a trajectories.zarr store")
        sessions.append((session_name(store), os.path.realpath(store)))
    pool = []
    for name, store in natsort.natsorted(sessions):
        for traj in trajectory_names(store):
            pool.append((name, traj, os.path.join(store, traj)))
    return pool


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sessions", nargs="+", required=True, help="session directories or stores")
    ap.add_argument("--out", required=True, help="output directory (must not exist yet)")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--test-count", type=int, default=10, help="evenly spaced test trajectories")
    group.add_argument("--test-trajs", default="", help="explicit test set: session/traj_K,...")
    ap.add_argument("--dry-run", action="store_true", help="print the split, write nothing")
    args = ap.parse_args()

    pool = pooled_trajectories(args.sessions)
    if len(pool) < 2:
        raise SystemExit(f"need at least 2 trajectories, found {len(pool)}")
    if args.test_trajs:
        wanted = {t.strip() for t in args.test_trajs.split(",") if t.strip()}
        test = {i for i, (s, t, _) in enumerate(pool) if f"{s}/{t}" in wanted}
        if len(test) != len(wanted):
            found = {f"{pool[i][0]}/{pool[i][1]}" for i in test}
            raise SystemExit(f"--test-trajs not in the pool: {sorted(wanted - found)}")
    else:
        n_test = max(1, min(args.test_count, len(pool) - 1))
        test = set(np.linspace(0, len(pool) - 1, n_test).round().astype(int).tolist())

    manifest = {
        "sessions": sorted({s for s, _, _ in pool}),
        "n_train": len(pool) - len(test),
        "n_test": len(test),
        "map": {},
    }
    for i, (session, traj, target) in enumerate(pool):
        split = "test" if i in test else "train"
        manifest["map"][f"traj_{i}"] = {
            "split": split,
            "session": session,
            "orig": traj,
            "target": target,
        }
    print(
        f"{len(pool)} trajectories from {len(manifest['sessions'])} sessions -> "
        f"train {manifest['n_train']}, test {manifest['n_test']}"
    )
    for name, rec in manifest["map"].items():
        if rec["split"] == "test":
            print(f"  test {name:>9} <- {rec['session']}/{rec['orig']}")
    if args.dry_run:
        return

    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} already exists; choose a new --out")
    for split in ("train", "test"):
        zarr.open_group(os.path.join(args.out, split, "trajectories.zarr"), mode="w")
    for name, rec in manifest["map"].items():
        os.symlink(rec["target"], os.path.join(args.out, rec["split"], "trajectories.zarr", name))
    with open(os.path.join(args.out, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
