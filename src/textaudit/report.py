"""Turn raw result CSVs into Markdown tables and figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _pm(mean: float, std: float, digits: int = 3) -> str:
    if pd.isna(mean):
        return "n/a"
    return f"{mean:.{digits}f}" if pd.isna(std) else f"{mean:.{digits}f} ± {std:.{digits}f}"


def _pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{100 * x:.1f}%"


def prep_section(df: pd.DataFrame, figures: Path) -> str:
    agg = df.groupby(["config", "model"])[["accuracy", "macro_f1"]].agg(["mean", "std"])
    base = {m: agg.loc[("base", m), ("accuracy", "mean")] for m in df["model"].unique() if ("base", m) in agg.index}
    rows = []
    for (config, model), r in agg.iterrows():
        delta = r[("accuracy", "mean")] - base.get(model, np.nan)
        rows.append([
            config, model,
            _pm(r[("accuracy", "mean")], r[("accuracy", "std")]),
            _pm(r[("macro_f1", "mean")], r[("macro_f1", "std")]),
            "—" if config == "base" else f"{delta:+.3f}",
        ])
    table = _md_table(["config", "model", "accuracy", "macro-F1", "Δ accuracy vs base"], rows)

    models = sorted(df["model"].unique())
    configs = list(dict.fromkeys(["base"] + sorted(set(df["config"]) - {"base"})))
    x = np.arange(len(configs))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, m in enumerate(models):
        means = [agg.loc[(c, m), ("accuracy", "mean")] if (c, m) in agg.index else np.nan for c in configs]
        stds = [agg.loc[(c, m), ("accuracy", "std")] if (c, m) in agg.index else 0 for c in configs]
        ax.bar(x - 0.4 + width * (i + 0.5), means, width, yerr=np.nan_to_num(stds), capsize=3, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha="right")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(max(0.0, np.nanmin(df["accuracy"]) - 0.05), min(1.0, df["accuracy"].max() + 0.03))
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "prep_ablation.png", dpi=150)
    plt.close(fig)
    return table + "\n\n![preprocessing ablation](figures/prep_ablation.png)"


def leakage_section(df: pd.DataFrame, figures: Path) -> str:
    agg = df.groupby(["dataset", "threshold"]).agg(
        frac=("frac_flagged", "mean"), acc_all=("acc_all", "mean"),
        acc_flagged=("acc_flagged", "mean"), acc_unflagged=("acc_unflagged", "mean"),
        median_sim=("median_max_similarity", "mean"),
    ).reset_index()
    rows = [
        [r["dataset"], f"{r['threshold']:.2f}", _pct(r["frac"]), _pct(r["acc_flagged"]),
         _pct(r["acc_unflagged"]), _pct(r["acc_all"])]
        for _, r in agg.iterrows()
    ]
    table = _md_table(
        ["dataset", "similarity ≥", "test docs flagged", "accuracy on flagged", "accuracy on unflagged", "overall accuracy"],
        rows,
    )
    datasets = list(dict.fromkeys(agg["dataset"]))
    thresholds = sorted(agg["threshold"].unique())
    x = np.arange(len(datasets))
    width = 0.8 / len(thresholds)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, t in enumerate(thresholds):
        sub = agg[agg["threshold"] == t].set_index("dataset").reindex(datasets)
        ax.bar(x - 0.4 + width * (i + 0.5), 100 * sub["frac"], width, label=f"≥ {t:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha="right")
    ax.set_ylabel("% of test documents with a near-duplicate in train")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "leakage.png", dpi=150)
    plt.close(fig)
    return table + "\n\n![near-duplicate leakage](figures/leakage.png)"


def noise_section(df: pd.DataFrame, figures: Path) -> str:
    cols = ["auroc", "precision_at_k", "acc_noisy_labels", "acc_after_removal", "acc_clean_labels"]
    agg = df.groupby("noise_rate")[cols].agg(["mean", "std"])
    rows = [
        [f"{rate:.0%}"] + [_pm(r[(c, "mean")], r[(c, "std")]) for c in cols]
        for rate, r in agg.iterrows()
    ]
    table = _md_table(
        ["noise rate", "detection AUROC", "precision@k", "test acc (noisy labels)",
         "test acc (flagged removed)", "test acc (true labels)"],
        rows,
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    rates = agg.index.to_numpy() * 100
    for col, label in (("acc_clean_labels", "true labels"), ("acc_noisy_labels", "noisy labels"),
                       ("acc_after_removal", "flagged examples removed")):
        ax.errorbar(rates, agg[(col, "mean")], yerr=np.nan_to_num(agg[(col, "std")]), marker="o", capsize=3, label=label)
    ax.set_xlabel("injected label noise (%)")
    ax.set_ylabel("test accuracy (clean test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "noise.png", dpi=150)
    plt.close(fig)
    return table + "\n\n![label noise](figures/noise.png)"


def build_report(results_dir: Path) -> Path:
    results_dir = Path(results_dir)
    figures = results_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    parts = [
        "# Results summary", "",
        "Generated by `textaudit report`. All numbers are on held-out test data, mean ± standard deviation over seeds.", "",
    ]
    for title, fname, fn, blurb in (
        ("Preprocessing ablation", "prep.csv", prep_section,
         "One factor is changed at a time relative to `base`. `keep_metadata` keeps headers, footers and quoted text."),
        ("Near-duplicate leakage", "leakage.csv", leakage_section,
         "Test documents whose cosine similarity (TF-IDF) to some training document is at least the threshold."),
        ("Label noise", "noise.csv", noise_section,
         "Symmetric noise is injected into training labels only; flagged examples are the k least self-confident "
         "out-of-fold predictions, with k equal to the number of injected errors."),
    ):
        path = results_dir / fname
        parts += [f"## {title}", "", blurb, ""]
        parts += [fn(pd.read_csv(path), figures) if path.exists() else f"_{fname} not found; run `textaudit run`._", ""]
    out = results_dir / "summary.md"
    out.write_text("\n".join(parts))
    return out
