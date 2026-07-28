"""The calibration verdict must consume its own provenance.

`tune_fwd_dd_head.py` computes `is_real` and writes it to `trained_on`, and for a
long time `passed` ignored it — so a calibration fitted to data MANUFACTURED to hit
PRED_MEAN_TARGET / TRUE_MEAN_TARGET shipped in assets/forward_calibration_isotonic.json
as `passed: true`.

That was a tautology, not a result. `synthetic_data()` draws `true` around
TRUE_MEAN_TARGET, sets pred = 0.7*true + (PRED_MEAN_TARGET - 0.7*TRUE_MEAN_TARGET)
+ noise, then re-centres pred on PRED_MEAN_TARGET exactly. The reported
"bias_before 5.760%" is just 0.1137 - 0.0561 by construction. The artifact even
carries the give-away: ic_before 0.878 against ic_target 0.5066 — the synthetic
data does not have the IC it claims to model.

Tests are TWO-SIDED on purpose. A `passed` that is always False is exactly as
useless as one that is always True; only checking the synthetic case would be
satisfied by `"passed": False` hardcoded.

    python -m pytest tests/test_calibration_provenance.py -q
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "assets" / "forward_calibration_isotonic.json"
SRC = ROOT / "pipeline" / "tune_fwd_dd_head.py"


def entry():
    return json.loads(ASSET.read_text(encoding="utf-8"))


def test_asset_exists_and_declares_provenance():
    d = entry()
    assert "trained_on" in d, "no provenance field at all"
    assert "passed" in d and "bias_within_tolerance" in d, sorted(d)


def test_measurement_and_verdict_are_separate_fields():
    """The bias measurement must survive. Collapsing them would lose the fact that
    the isotonic fit does reduce bias -- that part is real arithmetic."""
    d = entry()
    assert isinstance(d["bias_within_tolerance"], bool)
    assert isinstance(d["passed"], bool)
    assert "passed_requires" in d, "the rule must be stated in the artifact"


def test_synthetic_training_cannot_report_passed():
    """The whole point. If trained_on is not 'real', passed must be False."""
    d = entry()
    if d["trained_on"] != "real":
        assert d["passed"] is False, (
            f"trained_on={d['trained_on']!r} but passed={d['passed']} -- a "
            "calibration fitted to manufactured data is claiming a verdict"
        )


def test_real_training_is_allowed_to_pass():
    """Anti-vacuity in the other direction: the rule must not be 'always False'.
    With real data AND bias in tolerance, passed must be True."""
    d = entry()
    if d["trained_on"] == "real" and d["bias_within_tolerance"]:
        assert d["passed"] is True, (
            "real data inside tolerance but passed=False -- the gate has become "
            "unconditionally negative, which reports nothing"
        )


def test_the_verdict_is_a_conjunction_not_a_copy():
    """passed == bias_within_tolerance AND is_real. Verified structurally so the
    rule cannot silently degrade into `passed = bias_within_tolerance` (the
    original defect) or into a constant."""
    src = SRC.read_text(encoding="utf-8")
    assert '"passed": bool(abs(metrics["bias_after"]) < BIAS_TOLERANCE and is_real)' in src, (
        "the passed expression no longer conjoins is_real"
    )
    assert '"bias_within_tolerance"' in src, "the measurement field was dropped"


def test_console_does_not_claim_pass_on_synthetic():
    """The console used to print a bare 'PASS' while the JSON said otherwise.
    Two surfaces, one fact."""
    src = SRC.read_text(encoding="utf-8")
    assert "NOT A PASS" in src, "the synthetic branch no longer disclaims a pass"
    assert "PASS: bias <1% achieved on REAL data" in src, (
        "the real branch no longer prints an unambiguous pass"
    )


def test_synthetic_targets_are_still_the_tautology_they_were():
    """Guards the REASON this matters. If the shipped entry is synthetic, its
    'before' numbers should equal the hardcoded targets -- that identity is the
    evidence the run proved nothing. If this ever fails on a synthetic entry the
    generator changed and the reasoning above needs re-checking."""
    d = entry()
    if d["trained_on"] == "real":
        return
    assert abs(d["pred_mean_before"] - 0.1137) < 1e-6, d["pred_mean_before"]
    assert abs(d["true_mean"] - 0.0561) < 1e-6, d["true_mean"]
    assert abs(d["bias_before"] - (0.1137 - 0.0561)) < 1e-6, d["bias_before"]
