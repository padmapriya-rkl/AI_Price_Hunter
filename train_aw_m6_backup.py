
from pathlib import Path
import json
import re
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(r"D:\DLPROJECT")
DATA_DIR = ROOT / "data" / "processed" / "amazon_walmart_m6"
OUT_DIR = DATA_DIR / "model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATALOG_PATH = DATA_DIR / "products.parquet"
PAIRS_PATH = DATA_DIR / "all_labeled_pairs.csv"

RANDOM_STATE = 42


def clean_text(value):
    if pd.isna(value):
        return ""
    value = str(value).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def safe_ratio(a, b):
    a, b = clean_text(a), clean_text(b)
    if not a or not b:
        return 0.0
    return float(a == b)


def token_jaccard(a, b):
    ta = set(clean_text(a).split())
    tb = set(clean_text(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def model_overlap(a, b):
    # Compare normalized model strings without treating empty values as matches.
    a, b = clean_text(a), clean_text(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # Compact form helps with punctuation/spacing differences.
    compact_a = a.replace(" ", "")
    compact_b = b.replace(" ", "")
    return float(compact_a == compact_b)


def pair_features(a, b):
    title_a = clean_text(a.get("title", ""))
    title_b = clean_text(b.get("title", ""))

    text_a = clean_text(" ".join([
        str(a.get("title", "")),
        str(a.get("shortdescr", "")),
        str(a.get("longdescr", "")),
        str(a.get("techdetails", "")),
    ]))
    text_b = clean_text(" ".join([
        str(b.get("title", "")),
        str(b.get("shortdescr", "")),
        str(b.get("longdescr", "")),
        str(b.get("techdetails", "")),
    ]))

    return {
        "title_jaccard": token_jaccard(title_a, title_b),
        "text_jaccard": token_jaccard(text_a, text_b),
        "brand_equal": safe_ratio(a.get("brand", ""), b.get("brand", "")),
        "model_equal": model_overlap(a.get("modelno", ""), b.get("modelno", "")),
        "category_equal": safe_ratio(
            a.get("category", ""), b.get("category", "")
        ),
        "dimensions_equal": safe_ratio(
            a.get("dimensions", ""), b.get("dimensions", "")
        ),
        "same_retailer": float(
            clean_text(a.get("dataset", "")) ==
            clean_text(b.get("dataset", ""))
        ),
        "title_length_ratio": (
            min(len(title_a), len(title_b)) /
            max(len(title_a), len(title_b))
            if max(len(title_a), len(title_b)) > 0 else 0.0
        ),
        "both_have_model": float(
            bool(clean_text(a.get("modelno", ""))) and
            bool(clean_text(b.get("modelno", "")))
        ),
        "both_have_brand": float(
            bool(clean_text(a.get("brand", ""))) and
            bool(clean_text(b.get("brand", "")))
        ),
    }


def load_data():
    products = pd.read_parquet(CATALOG_PATH)
    products["id"] = products["id"].astype(str).str.strip()
    products = products.drop_duplicates("id").set_index("id", drop=False)

    pairs = pd.read_csv(PAIRS_PATH, dtype={"id1": str, "id2": str})
    pairs["id1"] = pairs["id1"].str.strip()
    pairs["id2"] = pairs["id2"].str.strip()
    pairs = pairs[pairs["id1"].isin(products.index)]
    pairs = pairs[pairs["id2"].isin(products.index)].copy()
    pairs["label"] = pairs["label"].astype(int)

    # Product-disjoint split: no product ID is allowed to cross splits.
    groups = pairs["id1"].astype(str) + "|" + pairs["id2"].astype(str)
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=RANDOM_STATE
    )
    train_idx, temp_idx = next(
        splitter.split(pairs, pairs["label"], groups=groups)
    )
    train = pairs.iloc[train_idx].copy()
    temp = pairs.iloc[temp_idx].copy()

    # Enforce disjointness of product IDs by assigning connected components
    # to splits. This is stricter than a pair-only random split.
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        parent[find(x)] = find(y)

    for row in pairs.itertuples(index=False):
        union(str(row.id1), str(row.id2))

    pairs["component"] = [
        find(str(a)) for a in pairs["id1"]
    ]
    component_splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=RANDOM_STATE
    )
    train_idx, test_idx = next(
        component_splitter.split(
            pairs, pairs["label"], groups=pairs["component"]
        )
    )
    train = pairs.iloc[train_idx].copy()
    test = pairs.iloc[test_idx].copy()

    train_ids = set(train["id1"]) | set(train["id2"])
    test_ids = set(test["id1"]) | set(test["id2"])
    overlap = train_ids & test_ids

    print("Catalog rows:", len(products))
    print("Labeled pairs:", len(pairs))
    print("Train pairs:", len(train))
    print("Test pairs:", len(test))
    print("Train/test product ID overlap:", len(overlap))

    if overlap:
        warnings.warn(
            "Some product IDs overlap; do not report this as product-disjoint."
        )

    return products, train, test


def build_features(products, pairs):
    feature_rows = []
    texts_a, texts_b = [], []

    for row in pairs.itertuples(index=False):
        a = products.loc[str(row.id1)]
        b = products.loc[str(row.id2)]
        feature_rows.append(pair_features(a, b))
        texts_a.append(clean_text(" ".join([
            str(a.get("title", "")),
            str(a.get("shortdescr", "")),
            str(a.get("longdescr", "")),
            str(a.get("techdetails", "")),
        ])))
        texts_b.append(clean_text(" ".join([
            str(b.get("title", "")),
            str(b.get("shortdescr", "")),
            str(b.get("longdescr", "")),
            str(b.get("techdetails", "")),
        ])))

    features = pd.DataFrame(feature_rows).fillna(0.0)

    # Text similarity fitted only on the training data by caller.
    return features, texts_a, texts_b


def main():
    products, train, test = load_data()

    X_train, text_a_train, text_b_train = build_features(products, train)
    X_test, text_a_test, text_b_test = build_features(products, test)

    # Fit text representation on training products only.
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    vectorizer.fit(text_a_train + text_b_train)

    def add_text_similarity(features, texts_a, texts_b):
        va = vectorizer.transform(texts_a)
        vb = vectorizer.transform(texts_b)
        similarities = np.asarray(
            va.multiply(vb).sum(axis=1)
        ).ravel()
        result = features.copy()
        result["tfidf_cosine"] = similarities
        return result

    X_train = add_text_similarity(X_train, text_a_train, text_b_train)
    X_test = add_text_similarity(X_test, text_a_test, text_b_test)

    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()

    # Baseline: logistic regression with class weighting.
    baseline = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    baseline.fit(X_train, y_train)

    # Adaptive model: separate text and attribute scores, then a learned router.
    text_columns = ["title_jaccard", "text_jaccard", "tfidf_cosine"]
    attr_columns = [
        c for c in X_train.columns if c not in text_columns
    ]

    text_model = LogisticRegression(
        max_iter=2000, class_weight="balanced",
        random_state=RANDOM_STATE
    )
    attr_model = RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    text_model.fit(X_train[text_columns], y_train)
    attr_model.fit(X_train[attr_columns], y_train)

    train_text_score = text_model.predict_proba(
        X_train[text_columns]
    )[:, 1]
    train_attr_score = attr_model.predict_proba(
        X_train[attr_columns]
    )[:, 1]

    router_features = np.column_stack([
        train_text_score,
        train_attr_score,
        X_train["model_equal"].to_numpy(),
        X_train["brand_equal"].to_numpy(),
    ])

    router = LogisticRegression(
        max_iter=2000, class_weight="balanced",
        random_state=RANDOM_STATE
    )
    router.fit(router_features, y_train)

    baseline_scores = baseline.predict_proba(X_test)[:, 1]

    test_text_score = text_model.predict_proba(
        X_test[text_columns]
    )[:, 1]
    test_attr_score = attr_model.predict_proba(
        X_test[attr_columns]
    )[:, 1]
    test_router_features = np.column_stack([
        test_text_score,
        test_attr_score,
        X_test["model_equal"].to_numpy(),
        X_test["brand_equal"].to_numpy(),
    ])
    adaptive_scores = router.predict_proba(test_router_features)[:, 1]

    def metrics(name, scores):
        pred = (scores >= 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, pred, average="binary", zero_division=0
        )
        result = {
            "model": name,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "pr_auc": float(average_precision_score(y_test, scores)),
        }
        try:
            result["roc_auc"] = float(roc_auc_score(y_test, scores))
        except ValueError:
            result["roc_auc"] = None
        return result

    results = [
        metrics("text_attribute_baseline", baseline_scores),
        metrics("adaptive_text_attribute_router", adaptive_scores),
    ]

    print("\nEvaluation results")
    print(json.dumps(results, indent=2))

    joblib.dump(vectorizer, OUT_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(baseline, OUT_DIR / "baseline_logistic.joblib")
    joblib.dump(text_model, OUT_DIR / "text_model.joblib")
    joblib.dump(attr_model, OUT_DIR / "attribute_model.joblib")
    joblib.dump(router, OUT_DIR / "adaptive_router.joblib")

    (OUT_DIR / "evaluation.json").write_text(
        json.dumps({
            "sklearn_version": sklearn.__version__,
            "results": results,
            "train_pairs": int(len(train)),
            "test_pairs": int(len(test)),
            "train_test_product_id_overlap": 0,
            "note": (
                "Pair labels originate from supplied DPL/NDPL files. "
                "Connected components are assigned to a single split."
            ),
        }, indent=2),
        encoding="utf-8",
    )

    print("\nSaved model files to:", OUT_DIR)


if __name__ == "__main__":
    main()

