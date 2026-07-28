"""build_real_v4_exec must not invent an embedding matrix.

It used to write a random 32-d matrix to the CANONICAL DATA_DIR/embedding.npz when
the file was missing:

    embedding = np.random.randn(N, 32).astype(np.float32)

Nothing downstream can tell that from a real forward pass, and embedding.npz is
gitignored (.gitignore line 3), so the fabrication never appeared in a diff. It only
ever surfaced as "embeddings" in whatever consumed it.

The block is small and sits ~1000 lines into a script whose main() needs the full SEC
bundle, so these are AST-level checks over the module rather than a live run. Parsed,
never grepped -- the comment above the fix quotes the removed call, and a substring
check would match the prose. That mistake has been made five times in this codebase.

    python -m pytest tests/test_no_fabricated_embedding.py -q
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "pipeline" / "build_real_v4_exec.py"
TREE = ast.parse(SRC.read_text(encoding="utf-8"))


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


def _calls():
    return [n for n in ast.walk(TREE) if isinstance(n, ast.Call)]


def test_no_random_draw_anywhere_in_the_module():
    """The whole file, not just the old block -- a fabricator that moved is not fixed."""
    bad = []
    for call in _calls():
        name = _dotted(call.func)
        if ".random." in f".{name}." or name.endswith((".randn", ".default_rng")):
            bad.append(name)
    assert not bad, f"random draws in {SRC.name}: {bad}"


def test_no_savez_writes_an_embedding_key():
    """np.savez_compressed(embed_path, embedding=...) is the specific shape that
    poisoned the pipeline. Any savez passing an `embedding=` kwarg is suspect."""
    offenders = []
    for call in _calls():
        if _dotted(call.func).endswith(("savez", "savez_compressed")):
            for kw in call.keywords:
                if kw.arg == "embedding":
                    offenders.append(ast.dump(kw.value)[:80])
    assert not offenders, f"a savez still writes an `embedding=` array: {offenders}"


def test_embedding_status_is_recorded_in_meta():
    """Absence must be STATED. A consumer has to be able to distinguish 'we have
    embeddings' from 'we had none and declined to invent some' -- otherwise the
    honest behaviour is indistinguishable from the dishonest one downstream."""
    src = SRC.read_text(encoding="utf-8")
    assert "embedding_status" in src, "the status variable is gone"
    assert '"embedding_npz": embedding_status' in src, (
        "real_rows_meta no longer records embedding presence"
    )


def test_the_status_has_both_values():
    """Anti-vacuity. A status hardcoded to one string reports nothing."""
    src = SRC.read_text(encoding="utf-8")
    assert '"present" if embed_path.exists() else "absent (not fabricated)"' in src, (
        "embedding_status is no longer a conditional over the file's existence"
    )


def test_the_missing_branch_warns_and_does_not_write():
    """Structural: inside `if not embed_path.exists():` there must be no call that
    writes a file. Behaviour cannot reach this branch without the full SEC bundle."""
    target = None
    for node in ast.walk(TREE):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.UnaryOp)
            and isinstance(node.test.op, ast.Not)
            and "embed_path.exists" in _dotted(getattr(node.test.operand, "func", node.test.operand))
        ):
            target = node
            break
    assert target is not None, "could not find the `if not embed_path.exists():` branch"
    writers = []
    for call in ast.walk(target):
        if isinstance(call, ast.Call):
            name = _dotted(call.func)
            if name.endswith(("savez", "savez_compressed", "write_text", "write_bytes", "save")):
                writers.append(name)
    assert not writers, f"the missing-embedding branch writes: {writers}"
