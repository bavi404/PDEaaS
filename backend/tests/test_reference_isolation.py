"""Guard: the independent steady-state reference must never reach training, normalisation or model selection.

Only the solver definition (pdeflow/data/physics.py) and the metrics module (secondary analysis, separate key)
may mention it. Any new module that references the reference file or arrays fails this test.
"""

from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "pdeflow"
ALLOWED = {"data/physics.py", "evaluation/metrics.py"}
TOKENS = ("u_reference", "test_reference", "solve_steady_state")


def test_reference_only_in_allowed_modules():
    offenders = {}
    for p in PKG.rglob("*.py"):
        rel = p.relative_to(PKG).as_posix()
        found = [t for t in TOKENS if t in p.read_text()]
        if found and rel not in ALLOWED:
            offenders[rel] = found
    assert not offenders, offenders


def test_preprocessing_does_not_read_reference():
    txt = (PKG.parents[1] / "scripts/preprocess.py").read_text()
    assert not any(t in txt for t in TOKENS)
