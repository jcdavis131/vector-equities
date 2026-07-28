"""A verifier must never create its own subject.

`verify_trades_v6.check_train_matrix()` used to fabricate the matrix it was meant to
check when the file was missing -- np.random Z, np.random fwd_ret_6m / fwd_dd_6m,
sector hardcoded "Tech" -- write it to the CANONICAL DATA_DIR/train_matrix_v6.npz,
and then run the real checks against the file it had just invented. It returned
True, so main()'s `sys.exit(0 if all(...) else 1)` exited 0 and the run reported
VERIFIED. And because it used the real filename, a later run would pick the
fabricated matrix up as genuine.

Tests are behavioural where it matters (does it write a file? what does it return?)
and structural only for the thing behaviour cannot see (that no np.random call
remains in the function body).

    python -m pytest tests/test_verifier_never_fabricates.py -q
"""

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "pipeline" / "verify_trades_v6.py"


def _load(tmp_data_dir):
    """Import verify_trades_v6 with DATA_DIR pointed at an empty temp dir, so the
    'missing matrix' branch is the one under test."""
    spec = importlib.util.spec_from_file_location("_vt6", SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_vt6"] = mod
    spec.loader.exec_module(mod)
    mod.DATA_DIR = tmp_data_dir
    return mod


def _fn_source(name):
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(SRC.read_text(encoding="utf-8"), node)
    raise AssertionError(f"{name} not found in {SRC}")


def test_missing_matrix_returns_false(tmp_path):
    """The verdict. Missing input is not a pass."""
    mod = _load(tmp_path)
    assert mod.check_train_matrix() is False


def test_missing_matrix_creates_no_file(tmp_path):
    """The behaviour that matters most: nothing is written, least of all under the
    canonical name a later run would trust."""
    mod = _load(tmp_path)
    mod.check_train_matrix()
    written = list(tmp_path.rglob("*"))
    assert written == [], f"verifier created {[str(p) for p in written]}"
    assert not (tmp_path / "train_matrix_v6.npz").exists()


def _dotted(node):
    """'np.random.randn' for an Attribute/Name chain, else ''."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def test_the_function_contains_no_random_draw():
    """Structural, because behaviour cannot prove absence across every branch.

    PARSED, NOT GREPPED. The first version of this test did
    `assert "np.random" not in src` and FAILED -- on the comment directly above the
    fix, which quotes the very calls it removed. This repo's own convention is that
    comments quote the code they discuss, so a substring check counts prose as code.
    That mistake has now been made five times in this codebase; ast is the only
    honest way to ask "is this call still here".
    """
    tree = ast.parse(_fn_source("check_train_matrix").lstrip())
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _dotted(node.func)
            if (
                ".random." in f".{name}."
                or name.endswith((".randn", ".default_rng", ".rand"))
                or name in ("randn", "default_rng")
            ):
                found.append(name)
    assert not found, f"random draws back in check_train_matrix: {found}"


def test_no_bare_except_returns_true():
    """`except Exception: return True` meant fabrication FAILING still reported a
    pass. A failure path must never be the pass path."""
    src = _fn_source("check_train_matrix")
    tree = ast.parse(src.lstrip())
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Return):
                    val = stmt.value
                    assert not (
                        isinstance(val, ast.Constant) and val.value is True
                    ), "an except handler in check_train_matrix returns True"


def test_a_present_matrix_is_still_actually_checked(tmp_path):
    """Anti-vacuity. A check that returns False for everything reports nothing, so
    prove the present-file path still runs and can disagree: a matrix with the
    WRONG shape must not pass."""
    np = pytest.importorskip("numpy")
    mod = _load(tmp_path)
    bad = tmp_path / "train_matrix_v6.npz"
    np.savez_compressed(
        bad,
        Z=np.zeros((3, 4), dtype="float32"),
        mask=np.ones((3, 4), dtype="float32"),
    )
    assert bad.exists()
    result = mod.check_train_matrix()
    assert result is False, "a 3x4 matrix passed the shape check"
    # and it did not repair or replace the bad file
    assert bad.exists(), "the verifier deleted or rewrote its subject"
