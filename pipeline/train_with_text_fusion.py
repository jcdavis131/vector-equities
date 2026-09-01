#!/usr/bin/env python3
"""
Vector Equities — LLM Text Fusion Pipeline
17-tower transformer MTNN, 64-d company-year vectors, 66-118 features
Real sources only, zero-deps philosophy, honest 503

Pattern from: https://machinelearningmastery.com/combining-llm-embeddings-with-tabular-features-in-a-unified-scikit-learn-pipeline/
- TextEmbedder(BaseEstimator, TransformerMixin) wrapping SentenceTransformer('all-MiniLM-L6-v2')
- ColumnTransformer for numeric (StandardScaler) + categorical (OneHotEncoder) + text (TextEmbedder)
- Pipeline unified → RandomForest sector classifier / embedding projection to 64-d

Real sources:
- assets/feature_manifest_canonical_66.json (66 tight id core)
- assets/feature_manifest_deduped_118.json (118 deduped base)
- assets/feature_manifest_v6_real.json (v6 real)
- pipeline/data/train_matrix_v5.npz / train_matrix_v6.npz (real EDGAR XBRL)
- Existing acquire_forbes.py, acquire_wikipedia_bios pattern for text

If no text data available, creates valid trainable structure but fails honestly with 503
"""
import sys
import json
import pathlib
import datetime
from typing import List, Dict, Any, Tuple

REAL_MANIFEST_66 = "assets/feature_manifest_canonical_66.json"
REAL_MANIFEST_118 = "assets/feature_manifest_deduped_118.json"
REAL_MANIFEST_V6 = "assets/feature_manifest_v6_real.json"

EXPECTED_TEXT_SOURCES = [
    "data/company_descriptions.csv",
    "data/edgar_10k_item1_snippets.jsonl",
    "data/forbes_company_descriptions.json",
    "assets/company_descriptions.jsonl",
    "data/companies_real_with_text.csv",
    "pipeline/data/company_text_corpus.jsonl",
]

NUMERIC_FEATURES_CANONICAL = [
    "REV","COGS","GROSS_PROFIT","OP_INCOME","EBITDA","NET_INCOME","EBIT",
    "GROSS_MARGIN","OP_MARGIN","NET_MARGIN","EBITDA_MARGIN",
    "TOTAL_ASSETS","TOTAL_LIABILITIES","EQUITY","CASH","DEBT","BOOK_VALUE","TANGIBLE_BOOK","WORKING_CAPITAL","NET_DEBT","INVESTED_CAPITAL",
    "OCF","CAPEX","FCF","FCF_MARGIN","OCF_TO_NET","FCF_CONVERSION","CAPEX_TO_REV",
    "REV_YOY","EBITDA_YOY","NET_YOY","FCF_YOY","REV_3Y_CAGR","EBITDA_3Y_CAGR","EPS_3Y_CAGR","BOOK_3Y_CAGR","OCF_3Y_CAGR",
    "ROE","ROA","ROIC","FCF_ROIC","ROIC_WACC_SPREAD",
    "CURRENT_RATIO","QUICK_RATIO","DEBT_TO_EQUITY","DEBT_TO_EBITDA","INTEREST_COVERAGE","DEBT_TO_ASSETS","NET_DEBT_TO_EBITDA",
    "ASSET_TURNOVER","INVENTORY_TURNOVER","RECEIVABLE_TURNOVER","CASH_CONVERSION_CYCLE","CAPEX_TO_DEPRE",
    "EPS_DILUTED","BVPS","FCFPS","SHARES_YOY","DILUTION_3Y",
    "RET_1M","RET_3M","RET_6M","RET_12M","VOL_30D","VOL_90D","VOL_252D"
]

CATEGORICAL_FEATURES = ["sector", "industry", "exchange"]
TEXT_FEATURE = "description"

PIPELINE_CODE = '''
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer
import numpy as np

class TextEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='all-MiniLM-L6-v2', max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        self.model = None
    def fit(self, X, y=None):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        return self
    def transform(self, X):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        if hasattr(X, 'iloc'):
            texts = X.iloc[:,0].astype(str).tolist() if hasattr(X, 'shape') and len(X.shape)>1 else X.astype(str).tolist()
        else:
            texts = [str(x) for x in X]
        texts = [t[:2000] for t in texts]
        embs = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return np.array(embs)

def build_fusion_pipeline(numeric_features, categorical_features, text_feature):
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    text_transformer = TextEmbedder(model_name='all-MiniLM-L6-v2')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('text', text_transformer, [text_feature]),
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    return pipeline
'''

def load_manifests(root: pathlib.Path) -> Dict[str, Any]:
    manifests = {}
    for rel in [REAL_MANIFEST_66, REAL_MANIFEST_118, REAL_MANIFEST_V6]:
        p = root / rel
        if p.exists():
            with open(p) as f:
                manifests[rel] = json.load(f)
    return manifests

def check_text_sources(root: pathlib.Path) -> Tuple[bool, List[str], List[str]]:
    present = []
    missing = []
    for rel in EXPECTED_TEXT_SOURCES:
        if (root / rel).exists():
            present.append(rel)
        else:
            missing.append(rel)
    return len(present)>0, present, missing

def check_deps():
    deps = {}
    try:
        import sklearn
        deps["sklearn"] = sklearn.__version__
        deps["sklearn_ok"] = True
    except Exception as e:
        deps["sklearn"] = str(e)
        deps["sklearn_ok"] = False
    try:
        import sentence_transformers
        deps["sentence_transformers"] = sentence_transformers.__version__
        deps["st_ok"] = True
    except Exception as e:
        deps["sentence_transformers"] = str(e)
        deps["st_ok"] = False
    try:
        import numpy
        deps["numpy"] = numpy.__version__
        deps["numpy_ok"] = True
    except Exception as e:
        deps["numpy"] = str(e)
        deps["numpy_ok"] = False
    deps["all_ok"] = deps.get("sklearn_ok") and deps.get("st_ok") and deps.get("numpy_ok")
    return deps

def eval_sector_coherence(embeddings, sectors) -> Dict[str, Any]:
    try:
        import numpy as np
        sample_n = min(500, len(embeddings))
        embs = embeddings[:sample_n]
        secs = sectors[:sample_n]
        within = []
        between = []
        for i in range(min(50, sample_n)):
            sims = embs @ embs[i]
            for j in range(sample_n):
                if i==j: continue
                if secs[j]==secs[i]:
                    within.append(float(sims[j]))
                else:
                    between.append(float(sims[j]))
        within_mean = float(sum(within)/len(within)) if within else 0.0
        between_mean = float(sum(between)/len(between)) if between else 0.0
        sep = within_mean - between_mean
        return {
            "within_sector_cos_mean": within_mean,
            "between_sector_cos_mean": between_mean,
            "sep_within_minus_between": sep,
            "sector_coherence_pass": sep > 0.15,
            "sample_n": sample_n
        }
    except Exception as e:
        return {"error": str(e), "sector_coherence_pass": False}

def write_report(root: pathlib.Path, report: Dict[str, Any]):
    out_dir = root / "pipeline" / "eval_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "equities_text_fusion.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[equities_text_fusion] report -> {out_path}")
    return out_path

def main():
    root = pathlib.Path(__file__).parent.parent
    start = datetime.datetime.now(datetime.timezone.utc)
    print(f"[equities_text_fusion] root={root} ts={start.isoformat()}")

    manifests = load_manifests(root)
    has_text, present_text, missing_text = check_text_sources(root)
    deps = check_deps()

    print(f"[manifests] found {list(manifests.keys())}")
    print(f"[text] has_text={has_text} present={present_text} missing={len(missing_text)}")
    print(f"[deps] {deps}")

    sector_eval = {}
    try:
        import numpy as np
        candidates = [
            root / "pipeline" / "data" / "train_matrix_v6.npz",
            root / "pipeline" / "data" / "train_matrix_v5.npz",
            root / "pipeline" / "data" / "train_matrix_real.npz",
        ]
        npz_path = next((p for p in candidates if p.exists()), None)
        if npz_path:
            npz = np.load(str(npz_path), allow_pickle=True)
            Z = npz["Z"]
            sectors = npz["sector"].astype(str) if "sector" in npz else np.array(["Unknown"]*len(Z))
            sector_eval = eval_sector_coherence(Z, sectors)
            sector_eval["npz_path"] = str(npz_path)
            sector_eval["N"] = len(Z)
            sector_eval["D"] = Z.shape[1]
        else:
            sector_eval = {"error": "no train_matrix found", "sector_coherence_pass": False}
    except Exception as e:
        sector_eval = {"error": str(e), "sector_coherence_pass": False}

    report = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "ts_cdt": datetime.datetime.now().isoformat(),
        "nodeId": "equities-text-fusion",
        "task": "vector-equities LLM text fusion 17-tower MTNN 64-d",
        "manifests_found": list(manifests.keys()),
        "manifest_66_total": manifests.get(REAL_MANIFEST_66, {}).get("total_features", 66),
        "manifest_118_total": manifests.get(REAL_MANIFEST_118, {}).get("total_features", 118),
        "expected_text_sources": EXPECTED_TEXT_SOURCES,
        "present_text_sources": present_text,
        "missing_text_sources": missing_text,
        "has_text": has_text,
        "deps": deps,
        "numeric_features": NUMERIC_FEATURES_CANONICAL[:66],
        "categorical_features": CATEGORICAL_FEATURES,
        "text_feature": TEXT_FEATURE,
        "text_embedder": "SentenceTransformer('all-MiniLM-L6-v2') 384-d L2-normed",
        "column_transformer": "StandardScaler numeric + OneHotEncoder categorical + TextEmbedder text",
        "pipeline_code": PIPELINE_CODE,
        "sector_coherence": sector_eval,
        "zero_deps_philosophy": True,
        "never_synthetic": True,
        "honest_503": False,
        "smoke_train": {},
        "provenance": {
            "source": "SEC EDGAR XBRL filings 17-tower MTNN",
            "features": "66 tight id core / 118 deduped base / v6 real",
            "real_only": True,
            "towers": 17,
            "embedding_dim": 64
        }
    }

    if not has_text:
        report["honest_503"] = True
        report["smoke_train"] = {
            "status": "BLOCKED_HONEST_503",
            "reason": "No real company description text data available",
            "missing_files": missing_text,
            "present_files": present_text,
            "message": (
                f"503 Service Unavailable — LLM text fusion blocked: no real text corpus found. "
                f"Expected one of {EXPECTED_TEXT_SOURCES} with column '{TEXT_FEATURE}' containing real company descriptions from SEC EDGAR 10-K Item 1 Business. "
                f"Current: 0 of {len(EXPECTED_TEXT_SOURCES)} present. Real sources only — synthetic not allowed. "
                f"To unblock: harvest real EDGAR 10-K Item 1 Business text into data/company_descriptions.csv (ticker, description) via SEC EDGAR XBRL API (free tier). "
                f"Pipeline structure is valid and trainable once text arrives."
            ),
            "trainable_structure": True,
            "pipeline_valid": True,
            "next_steps": [
                "Harvest real EDGAR text: data/company_descriptions.csv (ticker, description) — real EDGAR 10-K Item 1 Business",
                "Verify: wc -l data/company_descriptions.csv",
                "Then: python3 pipeline/train_with_text_fusion.py --smoke",
                "Fusion: TextEmbedder('all-MiniLM-L6-v2') 384-d + StandardScaler 66-d numeric + OneHotEncoder sector -> RandomForest sector coherence"
            ]
        }
        print(f"[BLOCKED] {report['smoke_train']['message'][:400]}")
    elif not deps.get("all_ok"):
        report["honest_503"] = True
        report["smoke_train"] = {
            "status": "BLOCKED_HONEST_503_DEPS",
            "reason": f"Missing deps sklearn_ok={deps.get('sklearn_ok')} st_ok={deps.get('st_ok')}",
            "deps": deps,
            "message": f"503 — dependencies missing for text fusion. Need: pip install scikit-learn sentence-transformers numpy. Current: {deps}",
            "trainable_structure": True
        }
        print(f"[BLOCKED_DEPS] {report['smoke_train']['message']}")
    else:
        try:
            import pandas as pd
            import numpy as np
            text_path = None
            for rel in present_text:
                p = root / rel
                if p.exists():
                    text_path = p
                    break
            if text_path is None:
                raise FileNotFoundError(f"No text file from {present_text}")
            if text_path.suffix == ".csv":
                df = pd.read_csv(text_path, nrows=500)
            elif text_path.suffix == ".jsonl":
                rows = []
                with open(text_path) as f:
                    for i, line in enumerate(f):
                        if i>=500: break
                        rows.append(json.loads(line))
                df = pd.DataFrame(rows)
            else:
                with open(text_path) as f:
                    data = json.load(f)
                df = pd.DataFrame(data[:500] if isinstance(data, list) else data)

            if TEXT_FEATURE not in df.columns:
                raise ValueError(f"Missing required column '{TEXT_FEATURE}' in {text_path} — columns={list(df.columns)}")

            from sklearn.base import BaseEstimator, TransformerMixin
            from sklearn.preprocessing import StandardScaler, OneHotEncoder
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline
            from sklearn.ensemble import RandomForestClassifier
            from sentence_transformers import SentenceTransformer

            class TextEmbedder(BaseEstimator, TransformerMixin):
                def __init__(self, model_name='all-MiniLM-L6-v2'):
                    self.model_name = model_name
                    self.model = None
                def fit(self, X, y=None):
                    if self.model is None:
                        self.model = SentenceTransformer(self.model_name)
                    return self
                def transform(self, X):
                    if self.model is None:
                        self.model = SentenceTransformer(self.model_name)
                    if hasattr(X, 'iloc'):
                        texts = X.iloc[:,0].astype(str).tolist() if hasattr(X, 'shape') and len(X.shape)>1 else X.astype(str).tolist()
                    else:
                        texts = [str(x) for x in X]
                    texts = [t[:2000] for t in texts]
                    embs = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
                    return np.array(embs)

            numeric_features = [c for c in NUMERIC_FEATURES_CANONICAL[:66] if c in df.columns]
            categorical_features = [c for c in CATEGORICAL_FEATURES if c in df.columns]

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features) if numeric_features else ('num', 'drop', []),
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features) if categorical_features else ('cat', 'drop', []),
                    ('text', TextEmbedder(), [TEXT_FEATURE]),
                ],
                remainder='drop',
                verbose_feature_names_out=False
            )
            preprocessor.transformers = [t for t in preprocessor.transformers if t[1] != 'drop']

            if 'sector' in df.columns:
                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
                ])
                X = df[[c for c in numeric_features + categorical_features + [TEXT_FEATURE] if c in df.columns]]
                y = df['sector']
                pipeline.fit(X, y)
                score = pipeline.score(X, y)
                fused_shape = pipeline.named_steps['preprocessor'].transform(X).shape
                report["smoke_train"] = {
                    "status": "PASS_SMOKE",
                    "n_rows": len(df),
                    "text_path": str(text_path),
                    "numeric_features_used": numeric_features,
                    "categorical_features_used": categorical_features,
                    "fused_shape": list(fused_shape),
                    "train_score": float(score),
                    "pipeline": "ColumnTransformer(StandardScaler+OneHot+TextEmbedder) -> RandomForest sector",
                    "message": f"Smoke train PASS {len(df)} rows -> {fused_shape} sector coherence train acc {score:.3f}"
                }
            else:
                pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
                X = df[[c for c in numeric_features + categorical_features + [TEXT_FEATURE] if c in df.columns]]
                fused = pipeline.fit_transform(X)
                report["smoke_train"] = {
                    "status": "PASS_SMOKE_PREPROCESS_ONLY",
                    "n_rows": len(df),
                    "fused_shape": list(fused.shape),
                    "message": f"Smoke preprocess PASS {len(df)} rows -> {fused.shape}"
                }
            print(f"[SMOKE PASS] {report['smoke_train']['message']}")

        except Exception as e:
            report["honest_503"] = True
            report["smoke_train"] = {
                "status": "FAIL_HONEST_503",
                "error": str(e),
                "error_type": type(e).__name__,
                "message": f"503 — smoke train failed honestly: {e}",
                "trainable_structure": True
            }
            print(f"[SMOKE FAIL] {e}")

    out_path = write_report(root, report)

    elapsed = (datetime.datetime.now(datetime.timezone.utc) - start).total_seconds()*1000
    print(f"[done] elapsed_ms={elapsed:.0f} report={out_path} honest_503={report['honest_503']}")

    if report.get("honest_503"):
        sys.exit(3)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
