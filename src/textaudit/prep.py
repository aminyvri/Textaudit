"""Preprocessing configurations, models, and the preprocessing ablation experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from .data import TextDataset


@dataclass(frozen=True)
class PrepConfig:
    lowercase: bool = True
    stop_words: bool = False
    min_df: int = 2
    ngram_range: tuple[int, int] = (1, 1)

    def vectorizer(self) -> TfidfVectorizer:
        return TfidfVectorizer(
            lowercase=self.lowercase,
            stop_words="english" if self.stop_words else None,
            min_df=self.min_df,
            ngram_range=self.ngram_range,
            sublinear_tf=True,
        )


# name -> (dataset variant, preprocessing). One factor is changed at a time relative to "base".
CONFIGS: dict[str, tuple[str, PrepConfig]] = {
    "base": ("clean", PrepConfig()),
    "keep_metadata": ("metadata", PrepConfig()),
    "no_lowercase": ("clean", PrepConfig(lowercase=False)),
    "stopwords": ("clean", PrepConfig(stop_words=True)),
    "min_df_1": ("clean", PrepConfig(min_df=1)),
    "min_df_5": ("clean", PrepConfig(min_df=5)),
    "bigrams": ("clean", PrepConfig(ngram_range=(1, 2))),
}


def make_model(name: str, seed: int = 0):
    if name == "linear_svm":
        return LinearSVC(C=1.0, random_state=seed)
    if name == "complement_nb":
        return ComplementNB(alpha=0.3)
    raise ValueError(f"Unknown model {name!r}")


MODEL_NAMES = ("linear_svm", "complement_nb")


def split_indices(labels, seed: int, test_size: float = 0.2):
    """Stratified train/test split; returns index arrays."""
    idx = np.arange(len(labels))
    return train_test_split(idx, test_size=test_size, stratify=labels, random_state=seed)


def run_prep_ablation(
    variants: dict[str, TextDataset],
    seeds,
    configs: dict[str, tuple[str, PrepConfig]] | None = None,
    models=MODEL_NAMES,
    verbose: bool = True,
) -> pd.DataFrame:
    """Train each (config, model) on a stratified 80/20 split, once per seed.

    The vectorizer is fitted on the training texts only.
    """
    configs = configs or CONFIGS
    rows = []
    for seed in seeds:
        for cfg_name, (variant, prep) in configs.items():
            if variant not in variants:
                continue
            ds = variants[variant]
            tr, te = split_indices(ds.labels, seed)
            vec = prep.vectorizer()
            X_tr = vec.fit_transform([ds.texts[i] for i in tr])
            X_te = vec.transform([ds.texts[i] for i in te])
            for model_name in models:
                clf = make_model(model_name, seed).fit(X_tr, ds.labels[tr])
                pred = clf.predict(X_te)
                row = {
                    "dataset": ds.name,
                    "config": cfg_name,
                    "model": model_name,
                    "seed": seed,
                    "accuracy": accuracy_score(ds.labels[te], pred),
                    "macro_f1": f1_score(ds.labels[te], pred, average="macro"),
                    "n_features": X_tr.shape[1],
                    "n_train": len(tr),
                    "n_test": len(te),
                }
                rows.append(row)
                if verbose:
                    print(f"[prep] seed={seed} {cfg_name:<14} {model_name:<13} acc={row['accuracy']:.4f}", flush=True)
    return pd.DataFrame(rows)
