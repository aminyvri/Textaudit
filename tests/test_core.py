import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from helpers import synthetic_dataset
from textaudit.data import load_csv
from textaudit.leakage import leakage_rows, max_similarity
from textaudit.noise import inject_symmetric_noise, noise_rows, self_confidence
from textaudit.prep import CONFIGS, PrepConfig, run_prep_ablation, split_indices


def test_split_is_stratified_and_disjoint():
    ds = synthetic_dataset()
    tr, te = split_indices(ds.labels, seed=1)
    assert set(tr).isdisjoint(te) and len(tr) + len(te) == len(ds)
    for c in range(ds.n_classes):
        assert abs((ds.labels[te] == c).mean() - 0.25) < 0.03


def test_prep_ablation_runs_and_metadata_leak_inflates_accuracy():
    clean = synthetic_dataset(seed=0, n_classes=6)
    # same documents, but with a token that reveals the label (like a newsgroup header would)
    leaky = synthetic_dataset(seed=0, n_classes=6, leak_token=True)
    df = run_prep_ablation({"clean": clean, "metadata": leaky}, seeds=[0, 1], verbose=False)
    assert set(df["config"]) == set(CONFIGS)
    assert (df["n_train"] + df["n_test"] == len(clean)).all()
    acc = df.groupby(["config", "model"])["accuracy"].mean()
    assert acc[("keep_metadata", "linear_svm")] >= acc[("base", "linear_svm")]
    assert acc[("keep_metadata", "linear_svm")] > 0.95


def test_max_similarity_matches_brute_force():
    docs = ["alpha beta gamma", "alpha beta delta", "xray yankee zulu", "alpha beta gamma"]
    X = TfidfVectorizer().fit_transform(docs)
    sims = max_similarity(X[[0, 2]], X[[1, 3]], chunk=1)
    dense = (X.toarray() @ X.toarray().T)
    assert sims[0] == pytest.approx(max(dense[0, 1], dense[0, 3]))
    assert sims[1] == pytest.approx(max(dense[2, 1], dense[2, 3]))
    assert sims[0] == pytest.approx(1.0)  # doc 0 equals doc 3


def test_leakage_detects_exact_duplicates():
    plain = synthetic_dataset(seed=1, n_duplicates=0)
    dup = synthetic_dataset(seed=1, n_duplicates=120)
    flagged_plain = leakage_rows(plain, seed=0, thresholds=(0.99,))["frac_flagged"].iloc[0]
    flagged_dup = leakage_rows(dup, seed=0, thresholds=(0.99,))["frac_flagged"].iloc[0]
    assert flagged_dup > flagged_plain + 0.1


def test_noise_injection_flips_exactly_the_requested_fraction():
    y = np.repeat(np.arange(5), 40)
    y_noisy, mask = inject_symmetric_noise(y, 0.2, 5, np.random.default_rng(0))
    assert mask.sum() == 40
    assert (y_noisy[mask] != y[mask]).all() and (y_noisy[~mask] == y[~mask]).all()
    same, no_mask = inject_symmetric_noise(y, 0.0, 5, np.random.default_rng(0))
    assert not no_mask.any() and (same == y).all()


def test_self_confidence_flags_noisy_labels_better_than_chance():
    ds = synthetic_dataset(n_per_class=100, seed=2)
    X = TfidfVectorizer(sublinear_tf=True).fit_transform(ds.texts)
    y_noisy, mask = inject_symmetric_noise(ds.labels, 0.15, ds.n_classes, np.random.default_rng(0))
    conf = self_confidence(X, y_noisy)
    k = mask.sum()
    precision = mask[np.argsort(conf)[:k]].mean()
    assert precision > 0.6  # chance would be 0.15


def test_noise_rows_shape_and_trend():
    ds = synthetic_dataset(n_per_class=100, seed=3)
    df = noise_rows(ds, seed=0, rates=(0.0, 0.3), verbose=False)
    assert list(df["noise_rate"]) == [0.0, 0.3]
    assert np.isnan(df.loc[0, "auroc"]) and df.loc[1, "auroc"] > 0.7
    assert df.loc[1, "acc_noisy_labels"] <= df.loc[1, "acc_clean_labels"] + 1e-9


def test_load_csv_encodes_labels(tmp_path):
    path = tmp_path / "toy.csv"
    pd.DataFrame({"text": ["a", "b", " ", "c"], "label": ["x", "y", "x", "x"]}).to_csv(path, index=False)
    ds = load_csv(str(path))
    assert len(ds) == 3 and ds.label_names == ["x", "y"]
    assert list(ds.labels) == [0, 1, 0]
