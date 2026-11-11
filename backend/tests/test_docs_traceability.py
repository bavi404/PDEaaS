"""Every headline number in docs/DATA.md and docs/PREPROCESSING.md must match the generated JSON reports.

Each entry: (doc, report, key path, formatter). The formatted report value must occur verbatim in the doc, so a
re-run that changes a statistic, or a hand-edited number, fails this test.
"""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def pct(nd):
    return lambda x: f"{100 * x:.{nd}f}%"


def fix(nd):
    return lambda x: f"{x:.{nd}f}"


def sci(nd):
    """2.9e-3 style (no zero-padded exponent), as written in the docs."""
    def f(x):
        m, e = f"{x:.{nd}e}".split("e")
        return f"{m}e{int(e)}"
    return f


def get(d, path):
    for k in path:
        d = d[k]
    return d


AUD = "reports/data_audit.json"
DS = "reports/downsampling_validation.json"
PRE = "reports/preprocessing_summary.json"
INS = "reports/data_inspection.json"
CT = ["convergence_test_split"]
DEV = CT + ["deviation_official_vs_reference"]
DUP = ["duplicate_groups", 0]
RB = ["region_breakdown"]
INT, BND, LAST = "interior (>=2 fine cells from edge)", "boundary band (outer 2 fine cells)", "last fine row+col only (extrapolated by stride-2)"

CASES = [
    # DATA.md: convergence (test split, exact solves)
    ("docs/DATA.md", AUD, DEV + ["median"], pct(2)),
    ("docs/DATA.md", AUD, DEV + ["mean"], pct(2)),
    ("docs/DATA.md", AUD, DEV + ["p90"], pct(1)),
    ("docs/DATA.md", AUD, DEV + ["p99"], pct(1)),
    ("docs/DATA.md", AUD, DEV + ["max"], pct(1)),
    ("docs/DATA.md", AUD, DEV + ["p75"], pct(2)),
    ("docs/DATA.md", AUD, CT + ["fraction_deviation_above", "0.01"], pct(1)),
    ("docs/DATA.md", AUD, CT + ["fraction_deviation_above", "0.05"], pct(1)),
    ("docs/DATA.md", AUD, CT + ["fraction_deviation_above", "0.1"], pct(1)),
    ("docs/DATA.md", AUD, CT + ["spearman_residual_vs_deviation"], fix(3)),
    ("docs/DATA.md", AUD, CT + ["spearman_frac_a1_vs_deviation"], lambda x: f"−{abs(x):.3f}"),
    ("docs/DATA.md", AUD, CT + ["mean_official_over_mean_reference"], fix(3)),
    ("docs/DATA.md", AUD, CT + ["deviation_by_fraction_a1", 0, "deviation_median"], pct(1)),
    ("docs/DATA.md", AUD, CT + ["deviation_by_fraction_a1", 4, "deviation_median"], pct(3)),
    ("docs/DATA.md", AUD, ["convergence_all_10000_estimated", "estimated_fraction_deviation_above", "0.01"], pct(1)),
    # DATA.md: duplicates
    ("docs/DATA.md", AUD, DUP + ["deviation_official_vs_reference", "median"], pct(1)),
    ("docs/DATA.md", AUD, DUP + ["deviation_official_vs_reference", "max"], pct(1)),
    ("docs/DATA.md", AUD, DUP + ["pairwise_rel_l2_between_official_targets_(norm_by_reference)", "median"], pct(1)),
    ("docs/DATA.md", AUD, DUP + ["pairwise_rel_l2_between_official_targets_(norm_by_reference)", "mean"], pct(1)),
    ("docs/DATA.md", AUD, DUP + ["rel_l2_to_group_mean", "mean"], pct(1)),
    ("docs/DATA.md", AUD, DUP + ["rel_l2_to_group_mean", "max"], pct(1)),
    ("docs/DATA.md", AUD, DUP + ["group_mean_deviation_vs_reference"], pct(1)),
    ("docs/DATA.md", AUD, DUP + ["official_minus_reference_sign", "frac_members_with_mean_above_reference"], pct(1)),
    ("docs/DATA.md", AUD, DUP + ["positions_in_shuffled_split", "first"], lambda x: f"{x:,}"),
    ("docs/DATA.md", AUD, DUP + ["positions_in_shuffled_split", "last"], lambda x: f"{x:,}"),
    ("docs/DATA.md", AUD, DUP + ["slowest_mode", "lambda1_continuum"], fix(2)),
    # DATA.md: inspection + per-split
    ("docs/DATA.md", INS, ["u_stats", "max"], fix(4)),
    ("docs/DATA.md", INS, ["u_stats", "mean"], fix(4)),
    ("docs/DATA.md", INS, ["pde_residual", "first200_rms_boundary_ring"], sci(1)),
    ("docs/DATA.md", INS, ["pde_residual", "first200_rms_boundary_ring_if_a_ghost_zero"], fix(1)),
    ("docs/DATA.md", PRE, ["files", "train", "u_mean"], fix(4)),
    ("docs/DATA.md", PRE, ["files", "test", "u_std"], fix(4)),
    ("docs/DATA.md", PRE, ["files", "val", "residual_frac_above_0.01"], pct(1)),
    # PREPROCESSING.md: normalisers
    ("docs/PREPROCESSING.md", PRE, ["normalizers", "a", "mean"], fix(6)),
    ("docs/PREPROCESSING.md", PRE, ["normalizers", "u", "std"], fix(6)),
    # PREPROCESSING.md: region breakdown
    ("docs/PREPROCESSING.md", DS, RB + [INT, "target_reconstruction_stride2", "mean"], pct(2)),
    ("docs/PREPROCESSING.md", DS, RB + [BND, "target_reconstruction_stride2", "mean"], pct(1)),
    ("docs/PREPROCESSING.md", DS, RB + [LAST, "target_reconstruction_stride2", "mean"], pct(1)),
    ("docs/PREPROCESSING.md", DS, RB + [INT, "information_loss_stride2", "mean"], pct(2)),
    ("docs/PREPROCESSING.md", DS, RB + [INT, "information_loss_avgpool", "mean"], pct(2)),
    ("docs/PREPROCESSING.md", DS, RB + [BND, "information_loss_stride2", "mean"], pct(2)),
    ("docs/PREPROCESSING.md", DS, RB + [BND, "target_reconstruction_stride2_max_abs"], fix(4)),
    ("docs/PREPROCESSING.md", DS, ["input_information_loss_rel_l2", "stride2", "mean"], pct(2)),
    ("docs/PREPROCESSING.md", DS, ["target_reconstruction_rel_l2", "stride2_then_linear_to_128_excluding_last_row_and_col", "mean"], pct(2)),
    ("docs/PREPROCESSING.md", DS, ["architecture_consistency_untrained_fno", "smooth_bandlimited_input", "mean"], pct(3)),
    ("docs/PREPROCESSING.md", DS, ["architecture_consistency_untrained_fno", "binary_permeability_input", "mean"], pct(3)),
    ("docs/PREPROCESSING.md", DS, ["spectral_energy_fraction", "a_above_k32", "mean"], pct(2)),
]


@pytest.mark.parametrize("doc,report,path,fmt", CASES, ids=[f"{c[0].split('/')[-1]}:{'.'.join(map(str, c[2]))}" for c in CASES])
def test_number_in_doc_matches_report(doc, report, path, fmt):
    value = get(json.loads((REPO / report).read_text()), path)
    text = (REPO / doc).read_text()
    assert fmt(value) in text, f"{doc} does not contain {fmt(value)!r} ({report}:{path})"
