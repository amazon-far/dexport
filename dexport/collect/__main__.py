"""Collect demonstrations: ``python -m dexport.collect --config configs/collect/<task>.yaml``.

The config names the hand model (Wuji 1 or Wuji 2), the hands and cameras in use and where the
episodes are saved; see :mod:`dexport.collect.config` and :mod:`dexport.collect.collector`.
"""

from __future__ import annotations

import argparse


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m dexport.collect",
        description="Collect demonstrations with Quest controllers and Wuji hands.",
    )
    ap.add_argument("--config", required=True, help="collection YAML (configs/collect/*.yaml)")
    args = ap.parse_args(argv)

    from dexport.collect.collector import Collector
    from dexport.collect.config import load_config

    Collector(load_config(args.config)).run()


if __name__ == "__main__":
    main()
