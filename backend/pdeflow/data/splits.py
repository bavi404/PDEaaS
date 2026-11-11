"""Deterministic train/val/test splits over the full PDEBench file.

A single seeded permutation is cut 80/10/10. Training on fewer samples uses a prefix of the
(already shuffled) train split, so the validation and test sets never change when the
training-set size changes, and no index can migrate between splits.

Group-aware: samples with byte-identical inputs (PDEBench beta=1.0 contains one group of 46
identical constant fields a = 0.1, see docs/DATA.md) are permuted and assigned as a unit, so an
input seen in training can never appear in validation or test. Split sizes are therefore within
one group size of the nominal fractions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Hashable, Sequence

import numpy as np


@dataclass
class SplitManifest:
    source_file: str
    source_md5: str
    n_total: int
    seed: int
    fractions: tuple[float, float, float]
    train: list[int] = field(repr=False)
    val: list[int] = field(repr=False)
    test: list[int] = field(repr=False)
    group_policy: str = "none"
    duplicate_groups: list[list[int]] = field(default_factory=list, repr=False)

    def subset(self, split: str, n: int | None = None) -> np.ndarray:
        idx = np.asarray(getattr(self, split), dtype=np.int64)
        if n is None:
            return idx
        if n > idx.size:
            raise ValueError(f"requested {n} samples from '{split}' which has {idx.size}")
        return idx[:n]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, path: str | Path) -> "SplitManifest":
        d = json.loads(Path(path).read_text())
        d["fractions"] = tuple(d["fractions"])
        m = cls(**d)
        m.validate()
        return m

    def validate(self) -> None:
        tr, va, te = (set(self.train), set(self.val), set(self.test))
        if len(tr) != len(self.train) or len(va) != len(self.val) or len(te) != len(self.test):
            raise ValueError("duplicate indices within a split")
        if tr & va or tr & te or va & te:
            raise ValueError("splits overlap")
        if len(tr | va | te) != self.n_total or min(tr | va | te) < 0 or max(tr | va | te) >= self.n_total:
            raise ValueError("splits do not partition range(n_total)")
        for g in self.duplicate_groups:
            homes = {name for name, s in (("train", tr), ("val", va), ("test", te)) if set(g) & s}
            if len(homes) != 1:
                raise ValueError(f"duplicate group spans splits {sorted(homes)}")


def make_splits(n_total: int, seed: int, fractions=(0.8, 0.1, 0.1), source_file: str = "", source_md5: str = "",
                groups: Sequence[Hashable] | None = None) -> SplitManifest:
    """Seeded split. ``groups[i]`` is a key for sample i; samples sharing a key stay together."""
    if abs(sum(fractions) - 1.0) > 1e-9 or any(f <= 0 for f in fractions):
        raise ValueError("fractions must be positive and sum to 1")
    if groups is None:
        groups = list(range(n_total))
    if len(groups) != n_total:
        raise ValueError("groups must have one key per sample")
    members: dict[Hashable, list[int]] = {}
    for i, g in enumerate(groups):
        members.setdefault(g, []).append(i)
    units = list(members.values())  # insertion order = order of first occurrence (deterministic)
    perm = np.random.default_rng(seed).permutation(len(units))
    targets = (fractions[0] * n_total, (fractions[0] + fractions[1]) * n_total)
    out: tuple[list[int], list[int], list[int]] = ([], [], [])
    filled = 0
    for k in perm:
        which = 0 if filled < targets[0] else (1 if filled < targets[1] else 2)
        out[which].extend(units[k])
        filled += len(units[k])
    dup = [sorted(u) for u in units if len(u) > 1]
    m = SplitManifest(
        source_file=source_file, source_md5=source_md5, n_total=n_total, seed=seed, fractions=tuple(fractions),
        train=out[0], val=out[1], test=out[2],
        group_policy="identical-input groups assigned as a unit" if dup else "none", duplicate_groups=dup,
    )
    m.validate()
    return m
