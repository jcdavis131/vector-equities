"""Hygiene: no ticker/FY leakage into tower inputs.

- FY embedding (12-d season_emb) must NOT be concatenated into tower X; it is
  fusion-only (EquitiesMTNN.encode: towers get xs, ms -> stack -> fusion(towers, season_ids, coverage)).
- Coverage scalar mean(mask) prevents zero-impute bias for temporally-skewed families (DEF14A FY2022+).
- Same-ticker adjacent-FY contrastive is honest only if year_norm excluded from X (career model pos_proj gated).
- No ticker string in feature spec (hoops player-split leak-free pattern reference).
"""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "pipeline"


def test_fy_embedding_excluded_from_tower_inputs():
    text = (PIPELINE_DIR / "model.py").read_text()
    # Tower forward signature: forward(self, x: Tensor, m: Tensor)
    assert "def forward(self, x: torch.Tensor, m: torch.Tensor)" in text
    # EquitiesMTNN.encode should call towers with xs[fam], ms[fam] only
    assert "self.towers[fam](xs[fam], ms[fam])" in text
    # Fusion receives season_ids separately, not via xs
    assert "def encode(self, xs, ms, season_ids)" in text
    assert "self.fusion(parts, season_ids, coverage)" in text
    # Ensure tower does NOT directly access season_emb
    # season_emb lives in fusion classes, not ResidualTower
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ResidualTower":
            src = ast.get_source_segment(text, node) or ""
            assert "season_emb" not in src, "ResidualTower must not see season_emb"
            assert "season_ids" not in src, "ResidualTower must not see season_ids"


def test_coverage_scalar_present():
    text = (PIPELINE_DIR / "model.py").read_text()
    # coverage = mean(mask) per family seen in encode
    assert "coverage = torch.stack" in text
    assert "mean(dim=-1)" in text
    assert "coverage.unsqueeze(-1)" in text, "tower_stack concat coverage scalar"
    # Comment about coverage scalar should exist in fusion classes
    assert "coverage scalar" in text.lower()


def test_no_ticker_in_feature_spec():
    sys.path.insert(0, str(PIPELINE_DIR))
    from feature_spec import ALL_FEATURES
    # ticker should never be a training feature
    assert "TICKER" not in ALL_FEATURES
    assert "ticker" not in [f.lower() for f in ALL_FEATURES]
    # FY embedding not a feature either
    assert "FY" not in ALL_FEATURES
    assert "YEAR" not in ALL_FEATURES
    # year_norm not a column in base spec
    low = [f.lower() for f in ALL_FEATURES]
    assert "year_norm" not in low


def test_career_year_norm_gated_not_in_tower_x():
    text = (PIPELINE_DIR / "model_career.py").read_text()
    # year_norm used only as positional additive in CausalCareerTransformer.pos_proj
    assert "pos_proj = nn.Linear(1, d_model)" in text
    assert "self.pos_proj(year_norm_seq)" in text
    # tower inputs are xs[fam], ms[fam] only, no year_norm in encode_timestep xs
    assert "def encode_timestep(self, xs, ms, time_enc)" in text
    assert "year_norm" not in (PIPELINE_DIR / "model.py").read_text().split("class EquitiesMTNN")[0]  # base MTNN has no year_norm


def test_train_mtnn_same_ticker_adjacent_FY_honest():
    text = (PIPELINE_DIR / "train_mtnn.py").read_text()
    # adjacent_pairs groups by ticker then FY diff ==1 — honest contrastive positive, not leak
    assert "def adjacent_pairs" in text
    assert "if y2 - y1 == 1" in text
    # family-drop train-only (hoops analogy)
    assert "drop_p" in text or "family-drop" in text.lower() or "batch_views" in text


def test_eval_excludes_same_ticker_for_cross_ticker():
    text = (PIPELINE_DIR / "eval_sector_coherence.py").read_text()
    assert "tickers ==" in text or "tickers != " in text or "tickers[i]" in text
    assert "cross_ticker" in text.lower()
    assert "block" in text  # chunked knn to avoid OOM, from shipped impl
    # ensure knn_indices supports tickers arg
    assert "def knn_indices" in text
    assert "tickers" in text
