"""Per-task training presets: ``python -m dexport.minbc.train <preset> ...``.

A preset is the shared recipe (the ``MinBCConfig`` defaults) plus the task's data
layout; ``--train-data`` / ``--test-data`` are required, and any other field can be overridden
on the command line. The state-as-action baseline is ``--data.hand-action-source state`` on any
preset.
"""

import tyro

from dexport.minbc.config import BIMANUAL_DATA_KEY, DataConfig, MinBCConfig, OptimConfig


def _preset(name, data, **fields):
    return MinBCConfig(
        output_name=name,
        train_data=tyro.MISSING,
        test_data=tyro.MISSING,
        data=DataConfig(**data),
        **fields,
    )


_RIGHT_WRIST = dict(hand_side="right", im_key=("right_rgb",))

PRESETS = {
    "pnp": (
        "Pick-and-place: Wuji 1 right hand, right wrist camera. (The state-as-action baseline of "
        "this task also uses --dp.down-dims 256 512 1024.)",
        _preset("pnp", _RIGHT_WRIST),
    ),
    "pnp_cotrain": (
        "Pick-and-place co-trained with in-the-wild data: --train-data LAB+WILD, half of every "
        "batch from each.",
        _preset(
            "pnp_cotrain",
            _RIGHT_WRIST,
            train_sample_mode="weighted",
            train_source_mix="0:0.5,1:0.5",
        ),
    ),
    "screwdriver": (
        "Screwdriver: Wuji 1 left hand, left wrist camera. --train-data joins 4 stores: source 0 "
        "fills half of every batch, sources 1-3 are pooled into the other half.",
        _preset(
            "screwdriver",
            dict(hand_side="left", im_key=("left_rgb",), im_stream=True),
            train_sample_mode="weighted",
            train_source_mix="0:0.5,1+2+3:0.5",
            optim=OptimConfig(num_workers=16),
        ),
    ),
    "drawer": (
        "Drawer stowing: Wuji 2 right hand, right wrist camera + left third-person camera.",
        _preset(
            "drawer",
            dict(hand_side="right", im_key=("right_rgb", "left_rgb"), im_stream=True),
            optim=OptimConfig(num_workers=16),
        ),
    ),
    "scan": (
        "Bimanual barcode scanning: two Wuji 2 hands (58-D actions), both wrist cameras, plus the "
        "relative pose between the hands.",
        _preset(
            "scan",
            dict(bimanual=True, data_key=BIMANUAL_DATA_KEY, im_key=("left_rgb", "right_rgb")),
        ),
    ),
}
