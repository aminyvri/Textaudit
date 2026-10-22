"""Loading text datasets: 20 Newsgroups (via scikit-learn) or any CSV with a text and a label column."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TextDataset:
    name: str
    texts: list[str]
    labels: np.ndarray          # integer class ids
    label_names: list[str]

    def __len__(self) -> int:
        return len(self.texts)

    @property
    def n_classes(self) -> int:
        return len(self.label_names)

    def subsample(self, n: int, seed: int = 0) -> "TextDataset":
        """Stratified-ish random subsample (used by --quick)."""
        if n >= len(self):
            return self
        idx = np.random.default_rng(seed).permutation(len(self))[:n]
        return TextDataset(
            self.name, [self.texts[i] for i in idx], self.labels[idx], self.label_names
        )


def load_20newsgroups(remove_metadata: bool, categories=None) -> TextDataset:
    """All 20 Newsgroups documents. Empty documents (e.g. after metadata removal) are dropped.

    With `remove_metadata=False` the raw posts keep their headers, footers and quoted text, which
    are known to leak information about the newsgroup (e.g. via e-mail domains and signatures).
    """
    from sklearn.datasets import fetch_20newsgroups

    remove = ("headers", "footers", "quotes") if remove_metadata else ()
    bunch = fetch_20newsgroups(subset="all", remove=remove, categories=categories, shuffle=False)
    texts = [t.strip() for t in bunch.data]
    keep = [i for i, t in enumerate(texts) if t]
    suffix = "clean" if remove_metadata else "with-metadata"
    return TextDataset(
        name=f"20ng-{suffix}",
        texts=[texts[i] for i in keep],
        labels=np.asarray(bunch.target)[keep],
        label_names=list(bunch.target_names),
    )


def load_csv(path: str, text_col: str = "text", label_col: str = "label", name: str | None = None) -> TextDataset:
    df = pd.read_csv(path)
    for col in (text_col, label_col):
        if col not in df.columns:
            raise KeyError(f"Column {col!r} not found in {path}; columns are {list(df.columns)}")
    df = df.dropna(subset=[text_col, label_col])
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""]
    codes, uniques = pd.factorize(df[label_col].astype(str), sort=True)
    return TextDataset(
        name=name or path.rsplit("/", 1)[-1].rsplit(".", 1)[0],
        texts=df[text_col].tolist(),
        labels=np.asarray(codes),
        label_names=[str(u) for u in uniques],
    )
