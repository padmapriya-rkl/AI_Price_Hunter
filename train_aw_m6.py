
from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(r"D:\DLPROJECT")
DATA_DIR = ROOT / "data" / "processed" / "amazon_walmart_m6"
OUT_DIR = DATA_DIR / "model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def clean(value):
    if pd.isna(value):
        return ""
    return re.sub(
        r"\s+", " ",
        re.sub(r"[^a-z0-9]+", " ", str(value).lower())
    ).strip()


def jaccard(a, b):
    a, b = set(clean(a).split()), set(clean(b).split())
    return len(a & b) / len(a | b) if a and b else 0.0


def equal_nonempty(a, b):
    a, b = clean(a), clean(b)
    return float(bool(a) and bool(b) and a == b)


def compact_equal(a, b):
    a, b = clean(a).replace(" ", ""), clean(b).replace(" ", "")
    return float(bool(a) and bool(b) and a == b)


def product_text(p):
    return clean(" ".join(
        str(p.get(c, ""))
        for c in ["title", "shortdescr", "longdescr", "techdetails"]
    ))


def pair_features(a, b):
    ta, tb = clean(a.get("title", "")), clean(b.get("title", ""))
    xa, xb = product_text(a), product_text(b)

    return {
        "title_jaccard": jaccard(ta, tb),
        "description_jaccard": jaccard(xa, xb),
        "brand_equal": equal_nonempty(a.get("brand", ""), b.get("brand", "")),
        "model_equal": compact_equal(a.get("modelno", ""), b.get("modelno", "")),
        "category_equal": equal_nonempty(
            a.get("category", ""), b.get("category", "")
        ),
        "dimensions_equal": equal_nonempty(
            a.get("dimensions", ""), b.get("dimensions", "")
        ),
        "same_retailer": float(
            clean(a.get("dataset", "")) == clean(b.get("dataset", ""))
        ),
        "both_have_model": float(
            bool(clean(a.get("modelno", "")))
            and bool(clean(b.get("modelno", "")))
        ),
        "both_have_brand": float(
            bool(clean(a.get("brand", "")))
            and bool(clean(b.get("brand", "")))
        ),
    }


def load_data():
    catalog_path = DATA_DIR / "products.parquet"
    pairs_path = DATA_DIR / "all_labeled_pairs.csv"

    products = pd.read_parquet(catalog_path)
    products["id"] = products["id"].astype(str).str.strip()
    products = products.drop_duplicates("id").set_index("id", drop=False)

    pairs = pd.read_csv(
        pairs_path, dtype={"id1": str, "id2": str}
    )
    pairs["id1"] = pairs["id1"].str.strip()
    pairs["id2"] = pairs["id2"].str.strip()
    pairs["label"] = pairs["label"].astype(int)
    pairs = pairs[
        pairs["id1"].isin(products.index)
        & pairs["id2"].isin(products.index)
    ].copy()

    # Build connected components: linked products must stay in one split.
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        parent[find(a)] = find(b)

    for r in pairs.itertuples(index=False):
        union(r.id1, r.id2)

    pairs["component"] = [find(x) for x in pairs["id1"]]

    # Split connected components into train, validation, and test.
    splitter1 = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=RANDOM_STATE
    )
    train_idx, remainder_idx = next(
        splitter1.split(pairs, pairs["label"], groups=pairs["component"])
    )
    train = pairs.iloc[train_idx].copy()
    remainder = pairs.iloc[remainder_idx].copy()

    splitter2 = GroupShuffleSplit(
        n_splits=1, test_size=0.50, random_state=RANDOM_STATE
    )
    val_idx, test_idx = next(
        splitter2.split(
            remainder, remainder["label"], groups=remainder["component"]
        )
    )
    validation = remainder.iloc[val_idx].copy()
    test = remainder.iloc[test_idx].copy()

    def ids(frame):
        return set(frame["id1"]) | set(frame["id2"])

    print("Catalog:", len(products))
    print("Pairs:", len(pairs))
    print("Train / validation / test:",
          len(train), len(validation), len(test))
    print("Positive pairs:",
          int((pairs["label"] == 1).sum()))
    print("Train-validation product overlap:",
          len(ids(train) & ids(validation)))
    print("Train-test product overlap:",
          len(ids(train) & ids(test)))
    print("Validation-test product overlap:",
          len(ids(validation) & ids(test)))

    if not len(train) or not len(validation) or not len(test):
        raise RuntimeError(
            "A split is empty. The pair graph may have too few "
            "independent connected components for this evaluation."
        )

    return products, train, validation, test


def make_features(products, pairs, vectorizer=None, fit=False):
    rows, text_a, text_b = [], [], []

    for r in pairs.itertuples(index=False):
        a, b = products.loc[r.id1], products.loc[r.id2]
        rows.append(pair_features(a, b))
        text_a.append(product_text(a))
        text_b.append(product_text(b))

    X = pd.DataFrame(rows).fillna(0.0)

    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )

    if fit:
        vectorizer.fit(text_a + text_b)

    va = vectorizer.transform(text_a)
    vb = vectorizer.transform(text_b)
    X["tfidf_cosine"] = np.asarray(
        va.multiply(vb).sum(axis=1)
    ).ravel()

    return X, vectorizer


def score_metrics(y, scores, threshold):
    pred = (scores >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )

    return {
        "threshold": float(threshold),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": (
            float(roc_auc_score(y, scores))
            if len(np.unique(y)) == 2 else None
        ),
    }


def select_threshold(y, scores):
    best = None
    # Select threshold using validation F1 only.
    for threshold in np.arange(0.05, 0.951, 0.01):
        result = score_metrics(y, scores, threshold)
        if best is None or result["f1"] > best["f1"]:
            best = result
    return best


def main():
    products, train, validation, test = load_data()

    X_train, vectorizer = make_features(
        products, train, fit=True
    )
    X_val, _ = make_features(products, validation, vectorizer)
    X_test, _ = make_features(products, test, vectorizer)

    y_train = train["label"].to_numpy()
    y_val = validation["label"].to_numpy()
    y_test = test["label"].to_numpy()

    # Candidate 1: text and attribute logistic regression.
    baseline = LogisticRegression(
        max_iter=2000, class_weight="balanced",
        random_state=RANDOM_STATE
    )
    baseline.fit(X_train, y_train)

    # Candidate 2: learned text and attribute scores, then a router.
    text_cols = [
        "title_jaccard", "description_jaccard", "tfidf_cosine"
    ]
    attr_cols = [c for c in X_train.columns if c not in text_cols]

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

    text_model.fit(X_train[text_cols], y_train)
    attr_model.fit(X_train[attr_cols], y_train)

    def router_input(X):
        return np.column_stack([
            text_model.predict_proba(X[text_cols])[:, 1],
            attr_model.predict_proba(X[attr_cols])[:, 1],
            X["model_equal"].to_numpy(),
            X["brand_equal"].to_numpy(),
        ])

    router = LogisticRegression(
        max_iter=2000, class_weight="balanced",
        random_state=RANDOM_STATE
    )
    router.fit(router_input(X_train), y_train)

    candidates = {
        "text_attribute_baseline": {
            "model": baseline,
            "scores": baseline.predict_proba(X_val)[:, 1],
        },
        "adaptive_text_attribute_router": {
            "model": router,
            "scores": router.predict_proba(router_input(X_val))[:, 1],
        },
    }

    validation_results = {}
    for name, candidate in candidates.items():
        validation_results[name] = select_threshold(
            y_val, candidate["scores"]
        )

    winner_name = max(
        validation_results,
        key=lambda name: validation_results[name]["f1"]
    )
    threshold = validation_results[winner_name]["threshold"]

    if winner_name == "text_attribute_baseline":
        test_scores = baseline.predict_proba(X_test)[:, 1]
        winner = baseline
    else:
        test_scores = router.predict_proba(router_input(X_test))[:, 1]
        winner = router

    test_result = score_metrics(y_test, test_scores, threshold)

    # Save the actual winning model with everything needed to use it.
    joblib.dump({
        "winner_name": winner_name,
        "winner_model": winner,
        "threshold": threshold,
        "vectorizer": vectorizer,
        "feature_columns": list(X_train.columns),
        "text_columns": text_cols,
        "attribute_columns": attr_cols,
        "text_model": text_model,
        "attribute_model": attr_model,
    }, OUT_DIR / "m6_model.joblib")

    report = {
        "sklearn_version": sklearn.__version__,
        "official_m6_model": winner_name,
        "selected_threshold": threshold,
        "selection_metric": "validation F1",
        "validation_results": validation_results,
        "final_test_metrics": test_result,
        "train_pairs": len(train),
        "validation_pairs": len(validation),
        "test_pairs": len(test),
        "note": (
            "Product-disjoint connected-component split. "
            "M6 currently uses text and attributes; no image branch "
            "is included because usable image URLs are not confirmed."
        ),
    }

    (OUT_DIR / "evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("\nValidation results:")
    print(json.dumps(validation_results, indent=2))
    print("\nOfficial M6 model:", winner_name)
    print("Selected threshold:", threshold)
    print("Final untouched test results:")
    print(json.dumps(test_result, indent=2))
    print("\nSaved:", OUT_DIR / "m6_model.joblib")


if __name__ == "__main__":
    main()

