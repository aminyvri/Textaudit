import pandas as pd

from helpers import synthetic_dataset
from textaudit.cli import build_parser, cmd_run, cmd_report


def test_end_to_end_cli_on_csv(tmp_path):
    ds = synthetic_dataset(n_per_class=60, n_classes=4, seed=4, n_duplicates=20)
    csv = tmp_path / "toy.csv"
    pd.DataFrame({"text": ds.texts, "label": [ds.label_names[i] for i in ds.labels]}).to_csv(csv, index=False)
    out = tmp_path / "results"
    args = build_parser().parse_args(["run", "--csv", str(csv), "--results-dir", str(out), "--seeds", "2"])
    cmd_run(args)
    for name in ("prep.csv", "leakage.csv", "noise.csv"):
        assert (out / name).exists()
    cmd_report(build_parser().parse_args(["report", "--results-dir", str(out)]))
    text = (out / "summary.md").read_text()
    for header in ("Preprocessing ablation", "Near-duplicate leakage", "Label noise"):
        assert header in text
    for fig in ("prep_ablation.png", "leakage.png", "noise.png"):
        assert (out / "figures" / fig).exists()
