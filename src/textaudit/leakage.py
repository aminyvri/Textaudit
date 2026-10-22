"""Near-duplicate leakage between train and test documents."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from .data import TextDataset
from .prep import PrepConfig, make_model, split_indices

DEFAULT_THRESHOLDS = (0.8, 0.9, 0.95)


def max_similarity(X_test, X_train, chunk: int = 500) -> np.ndarray:
    """Cosine similarity of each test row to its most similar training row.

    Rows must be L2-normalised (TfidfVectorizer does this by default), so the dot product is the cosine.
    Processed in chunks to keep memory bounded.
    """
    out = np.empty(X_test.shape[0], dtype=np.float64)
    X_train_t = X_train.T.tocsr()
    for start in range(0, X_test.shape[0], chunk):
        block = X_test[start : start + chunk] @ X_train_t
        out[start : start + chunk] = block.max(axis=1).toarray().ravel()
    return out


def _safe_acc(correct: np.ndarray, mask: np.ndarray) -> float:
    return float(correct[mask].mean()) if mask.any() else float("nan")


def leakage_rows(
    ds: TextDataset,
    seed: int,
    thresholds=DEFAULT_THRESHOLDS,
    prep: PrepConfig = PrepConfig(),
    model: str = "linear_svm",
) -> pd.DataFrame:
    """For one random split: how many test documents have a near-duplicate in train, and how do they score?"""
    tr, te = split_indices(ds.labels, seed)
    vec = prep.vectorizer()
    X_tr = vec.fit_transform([ds.texts[i] for i in tr])
    X_te = vec.transform([ds.texts[i] for i in te])
    sims = max_similarity(X_te, X_tr)
    pred = make_model(model, seed).fit(X_tr, ds.labels[tr]).predict(X_te)
    correct = pred == ds.labels[te]
    overall = float(accuracy_score(ds.labels[te], pred))

    rows = []
    for t in thresholds:
        flagged = sims >= t
        rows.append(
            {
                "dataset": ds.name,
                "seed": seed,
                "threshold": t,
                "n_test": len(te),
                "frac_flagged": float(flagged.mean()),
                "acc_all": overall,
                "acc_flagged": _safe_acc(correct, flagged),
                "acc_unflagged": _safe_acc(correct, ~flagged),
                "median_max_similarity": float(np.median(sims)),
            }
        )
    return pd.DataFrame(rows)
