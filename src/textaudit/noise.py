"""Label-noise injection and detection by out-of-fold self-confidence."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .data import TextDataset
from .prep import PrepConfig, split_indices

DEFAULT_RATES = (0.0, 0.05, 0.1, 0.2, 0.3)


def inject_symmetric_noise(y: np.ndarray, rate: float, n_classes: int, rng: np.random.Generator):
    """Flip a random `rate` fraction of labels to a different, uniformly chosen class."""
    y_noisy = y.copy()
    n_flip = int(round(rate * len(y)))
    mask = np.zeros(len(y), dtype=bool)
    if n_flip == 0:
        return y_noisy, mask
    idx = rng.choice(len(y), size=n_flip, replace=False)
    shift = rng.integers(1, n_classes, size=n_flip)  # 1..K-1 guarantees a different class
    y_noisy[idx] = (y[idx] + shift) % n_classes
    mask[idx] = True
    return y_noisy, mask


def make_probabilistic_model(seed: int = 0) -> LogisticRegression:
    return LogisticRegression(C=10.0, max_iter=300, random_state=seed)


def self_confidence(X, y_noisy: np.ndarray, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold probability the model assigns to each example's *given* label (low = suspicious)."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    classes = np.unique(y_noisy)
    probs = cross_val_predict(make_probabilistic_model(seed), X, y_noisy, cv=cv, method="predict_proba")
    col = {c: j for j, c in enumerate(classes)}
    cols = np.array([col[v] for v in y_noisy])
    return probs[np.arange(len(y_noisy)), cols]


def noise_rows(
    ds: TextDataset,
    seed: int,
    rates=DEFAULT_RATES,
    prep: PrepConfig = PrepConfig(),
    verbose: bool = True,
) -> pd.DataFrame:
    """For each noise rate: detection quality, and test accuracy when training on
    noisy labels, on labels with the flagged examples removed, and on the true labels.

    The test set is never corrupted. The TF-IDF vocabulary is fitted on training texts only
    (it uses no labels, so fitting it before cross-validation does not leak label information).
    """
    tr, te = split_indices(ds.labels, seed)
    vec = prep.vectorizer()
    X_tr = vec.fit_transform([ds.texts[i] for i in tr])
    X_te = vec.transform([ds.texts[i] for i in te])
    y_tr_true, y_te = ds.labels[tr], ds.labels[te]

    def test_acc(X, y):
        return float(accuracy_score(y_te, make_probabilistic_model(seed).fit(X, y).predict(X_te)))

    acc_clean_labels = test_acc(X_tr, y_tr_true)
    rows = []
    for rate in rates:
        rng = np.random.default_rng(seed * 1000 + int(round(rate * 100)))
        y_noisy, mask = inject_symmetric_noise(y_tr_true, rate, ds.n_classes, rng)
        row = {
            "dataset": ds.name, "seed": seed, "noise_rate": rate, "n_train": len(tr),
            "n_noisy": int(mask.sum()),
            "acc_noisy_labels": test_acc(X_tr, y_noisy),
            "acc_clean_labels": acc_clean_labels,
            "auroc": float("nan"), "precision_at_k": float("nan"), "acc_after_removal": float("nan"),
        }
        if mask.any():
            conf = self_confidence(X_tr, y_noisy, seed=seed)
            k = int(mask.sum())
            flagged = np.argsort(conf)[:k]          # the k least self-confident examples
            row["auroc"] = float(roc_auc_score(mask, 1.0 - conf))
            row["precision_at_k"] = float(mask[flagged].mean())
            keep = np.setdiff1d(np.arange(len(y_noisy)), flagged)
            row["acc_after_removal"] = test_acc(X_tr[keep], y_noisy[keep])
        else:
            row["acc_after_removal"] = row["acc_noisy_labels"]
        rows.append(row)
        if verbose:
            print(
                f"[noise] seed={seed} rate={rate:.2f} acc_noisy={row['acc_noisy_labels']:.4f} "
                f"acc_removed={row['acc_after_removal']:.4f} auroc={row['auroc']:.3f}",
                flush=True,
            )
    return pd.DataFrame(rows)
