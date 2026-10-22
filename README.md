# textaudit

Auditing text-classification benchmarks: how much do **preprocessing choices**, **near-duplicate leakage** and
**label noise** change what a model's accuracy appears to mean?

Reported accuracy on a text benchmark depends on decisions that are rarely documented: which parts of a document
are kept, how it is tokenised, whether the test set contains copies of training documents, and how clean the labels
are. This repository measures each of these effects on the 20 Newsgroups corpus (or on your own CSV) with simple,
transparent models, so the effect of the data decision is not confused with the effect of a complicated model.

It is an evaluation and data-quality study, not a new classifier.

## Experiments

All experiments use stratified 80/20 train/test splits repeated over several seeds. TF-IDF vocabularies are fitted on
training texts only. Models: linear SVM and Complement Naive Bayes (logistic regression for the noise experiment,
because it gives probabilities).

1. **Preprocessing ablation** (`src/textaudit/prep.py`).
   One factor at a time is changed relative to a base pipeline (metadata removed, lowercase, no stop-word removal,
   `min_df=2`, unigrams): keeping the raw headers/footers/quotes, no lowercasing, stop-word removal, `min_df` 1 or 5,
   and adding bigrams. Question: which decisions change test accuracy, and by how much?
2. **Near-duplicate leakage** (`src/textaudit/leakage.py`).
   For each test document, the cosine similarity of its TF-IDF vector to the most similar training document. Documents above a
   threshold (0.80, 0.90, 0.95) are flagged. Question: how many test documents have a near-copy in training, and do
   they score differently from the rest?
3. **Label noise** (`src/textaudit/noise.py`).
   Symmetric label noise (0–30%) is injected into the training labels only. Suspicious examples are found with
   out-of-fold *self-confidence*: the probability a cross-validated model assigns to each example's own label. Questions:
   how well does this detect the injected errors (AUROC, precision@k), and does removing the flagged examples recover
   accuracy on the clean test set?

## Quick start

```bash
git clone <this repository>
cd textaudit
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

make test        # unit tests, run on synthetic data (no download needed)
make quick       # 4000 documents, 2 seeds: fast end-to-end check
make all         # full 20 Newsgroups run, 3 seeds
```

Without `make`:

```bash
textaudit run                 # or: textaudit run --quick
textaudit report
```

20 Newsgroups is downloaded by scikit-learn on first use (internet needed once). To audit your own data:

```bash
textaudit run --csv path/to/data.csv --text-col text --label-col label
```

With a CSV only the `base`-style configurations are run (there is no metadata variant).

Outputs go to `results/`: `prep.csv`, `leakage.csv`, `noise.csv`, `summary.md` and `figures/*.png`.

## Results

> **TODO (fill in after running):** paste the main tables from `results/summary.md` and describe what you observe in
> a few sentences: which preprocessing choices mattered, how large the metadata effect is, what fraction of test
> documents have near-duplicates and how they score, and how well noise detection works at each noise rate. Report only
> what your own run produced.

## Limitations

- Linear models on TF-IDF only; conclusions may differ for neural or pretrained models.
- 20 Newsgroups is an old benchmark, chosen because its metadata leakage is well documented and it downloads without
  an account. Findings need not transfer to other corpora.
- Near-duplicate detection uses TF-IDF cosine similarity with fixed thresholds; paraphrases and translations are missed.
- Injected noise is symmetric and random. Real annotation errors are usually systematic, so detection results on real
  noise can be worse.
- Few seeds: standard deviations are indicative, not confidence intervals.
- Self-confidence detection is a simple baseline, not the full confident-learning method.

## Project layout

```
src/textaudit/
  data.py      20 Newsgroups and CSV loading
  prep.py      preprocessing configurations, models, ablation experiment
  leakage.py   near-duplicate detection between train and test
  noise.py     noise injection, self-confidence detection, effect of removal
  report.py    Markdown tables and figures
  cli.py       `textaudit` command
tests/         unit tests on synthetic data, including an end-to-end CLI test
```

## Reproducibility

Seeds are `0 .. seeds-1` and control the splits and the models. Library versions are in `requirements.txt`; exact numbers
can differ slightly across scikit-learn versions and across 20 Newsgroups downloads.

## Development note

The code in this repository was developed with AI assistance (Claude). The study design, the runs, and the analysis
of the results are by the repository author.

## License

MIT, see `LICENSE`.
