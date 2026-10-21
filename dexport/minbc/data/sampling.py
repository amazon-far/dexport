"""Train-window sampling across the '+'-joined ``train_data`` sources (co-training mixes).

``train_sample_mode``: ``concat`` shuffles all windows uniformly (bigger sources contribute
more); ``weighted`` gives each group of sources in ``train_source_mix`` an explicit share of the
draws (``"0:0.5,1:0.5"`` is the 50/50 co-training mix; ``"0:0.5,1+2+3:0.5"`` pools sources 1-3
into one half); ``balanced`` gives every source an equal share (``weighted`` with equal weights,
without spelling out a mix). With a single source, as in saved single-source configs that set
it, ``balanced`` is the ``concat`` shuffle.
"""

import numpy as np
import torch


def parse_source_mix(spec: str, num_sources: int) -> list[tuple[list[int], float]]:
    """Parse ``"<srcs>:<weight>,..."`` into ``[(source_ids, weight), ...]``, weights summing to 1.

    ``<srcs>`` is one or more 0-based positions in the '+'-split of ``train_data`` joined by '+'.
    Every source must appear exactly once: a silently re-mixed co-train has no other symptom.
    """
    if not str(spec or "").strip():
        raise ValueError(
            "train_sample_mode='weighted' requires --train-source-mix "
            '(e.g. "0:0.4,1:0.3,2:0.3" or "0+1:0.7,2:0.3").'
        )
    groups = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.count(":") != 1:
            raise ValueError(
                f"train_source_mix group {tok!r} is not '<srcs>:<weight>' (one colon per group)."
            )
        srcs_s, w_s = tok.split(":")
        try:
            w = float(w_s)
        except ValueError:
            raise ValueError(f"train_source_mix group {tok!r}: weight {w_s!r} is not a number.")
        if not (w > 0) or not np.isfinite(w):
            raise ValueError(f"train_source_mix group {tok!r}: weight must be finite and > 0.")
        ids = []
        for s in srcs_s.split("+"):
            s = s.strip()
            if not s.isdigit():
                raise ValueError(
                    f"train_source_mix group {tok!r}: {s!r} is not a 0-based source index."
                )
            ids.append(int(s))
        groups.append((ids, w))
    if not groups:
        raise ValueError(f"train_source_mix {spec!r} parsed to no groups.")

    flat = [i for ids, _ in groups for i in ids]
    dupes = sorted({i for i in flat if flat.count(i) > 1})
    if dupes:
        raise ValueError(
            f"train_source_mix {spec!r}: source(s) {dupes} listed more than once "
            f"(a source belongs to exactly one group)."
        )
    bad = sorted(i for i in flat if not (0 <= i < num_sources))
    if bad:
        raise ValueError(
            f"train_source_mix {spec!r}: source index(es) {bad} out of range for "
            f"{num_sources} train_data source(s) (valid 0..{num_sources - 1})."
        )
    missing = sorted(set(range(num_sources)) - set(flat))
    if missing:
        raise ValueError(
            f"train_source_mix {spec!r} omits source(s) {missing}; every one of the "
            f"{num_sources} train_data sources must be listed."
        )
    total = float(sum(w for _, w in groups))
    return [(ids, w / total) for ids, w in groups]


def balanced_sample_weights(source_ids: np.ndarray, num_sources: int) -> np.ndarray:
    """Per-window weights giving every source with windows the same total mass."""
    counts = np.bincount(source_ids, minlength=num_sources).astype(np.float64)
    weights = np.zeros(len(source_ids), dtype=np.float64)
    for s in range(num_sources):
        if counts[s] > 0:
            weights[source_ids == s] = 1.0 / counts[s]
    return weights


def weighted_sample_weights(source_ids: np.ndarray, groups) -> np.ndarray:
    """Per-window weights giving each group its mass, spread uniformly over the group's windows."""
    masks = []
    for ids, w in groups:
        mask = np.isin(source_ids, np.asarray(ids, dtype=np.int64))
        n = int(mask.sum())
        if n == 0:
            raise ValueError(
                f"train_source_mix group {ids} (weight {w:.4f}) has no windows; the requested "
                f"mix cannot be realized. Check the train_data order and that each source has "
                f"trajectories."
            )
        masks.append((mask, n, float(w)))
    total = sum(w for _, _, w in masks)
    weights = np.zeros(len(source_ids), dtype=np.float64)
    for mask, n, w in masks:
        weights[mask] = (w / total) / n
    return weights


def make_train_sampler(config, dataset):
    """A ``WeightedRandomSampler`` for ``weighted`` and multi-source ``balanced``, else None
    (a plain shuffle)."""
    mode = config.train_sample_mode
    source_ids, num_sources = dataset.source_ids, dataset.num_sources
    if mode == "weighted":
        groups = parse_source_mix(config.train_source_mix, num_sources)
        weights = weighted_sample_weights(source_ids, groups)
        counts = np.bincount(source_ids, minlength=num_sources)
        desc = ", ".join(
            "+".join(map(str, ids)) + f"={100 * w:.1f}% ({sum(int(counts[i]) for i in ids):,} win)"
            for ids, w in groups
        )
        print(f"[sampler] weighted over {num_sources} sources: {desc}")
    elif mode == "balanced" and num_sources > 1:
        weights = balanced_sample_weights(source_ids, num_sources)
        print(f"[sampler] balanced: {num_sources} sources, equal probability each")
    elif mode in ("concat", "balanced"):
        return None
    else:
        raise ValueError(f"unknown train_sample_mode {mode!r} (concat, balanced or weighted)")
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
    )
