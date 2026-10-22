"""Command line interface: run the three experiments, then build the report."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data import TextDataset, load_20newsgroups, load_csv
from .leakage import DEFAULT_THRESHOLDS, leakage_rows
from .noise import DEFAULT_RATES, noise_rows
from .prep import run_prep_ablation


def load_variants(args) -> dict[str, TextDataset]:
    """Dataset variants keyed by role: 'clean' (used by most experiments) and, for 20NG, 'metadata'."""
    if args.csv:
        ds = load_csv(args.csv, args.text_col, args.label_col)
        variants = {"clean": ds}
    else:
        variants = {"clean": load_20newsgroups(True), "metadata": load_20newsgroups(False)}
    if args.quick:
        variants = {k: v.subsample(4000, seed=0) for k, v in variants.items()}
    for key, ds in variants.items():
        print(f"[data] {key}: {ds.name}, {len(ds)} documents, {ds.n_classes} classes", flush=True)
    return variants


def cmd_run(args) -> None:
    out = args.results_dir
    out.mkdir(parents=True, exist_ok=True)
    variants = load_variants(args)
    seeds = list(range(2 if args.quick else args.seeds))
    wanted = set(args.experiments)

    if "prep" in wanted:
        run_prep_ablation(variants, seeds).to_csv(out / "prep.csv", index=False)
    if "leakage" in wanted:
        frames = [leakage_rows(ds, seed, DEFAULT_THRESHOLDS) for ds in variants.values() for seed in seeds]
        pd.concat(frames, ignore_index=True).to_csv(out / "leakage.csv", index=False)
    if "noise" in wanted:
        clean = variants["clean"]
        frames = [noise_rows(clean, seed, DEFAULT_RATES) for seed in seeds]
        pd.concat(frames, ignore_index=True).to_csv(out / "noise.csv", index=False)
    print(f"[run] results written to {out}")


def cmd_report(args) -> None:
    from .report import build_report

    print(f"[report] wrote {build_report(args.results_dir)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="textaudit", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run the experiments")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--csv", help="use your own CSV instead of 20 Newsgroups")
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--experiments", nargs="+", default=["prep", "leakage", "noise"],
                   choices=["prep", "leakage", "noise"])
    p.add_argument("--quick", action="store_true", help="4000-document subsample and 2 seeds")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("report", help="build results/summary.md and figures")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.set_defaults(func=cmd_report)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
