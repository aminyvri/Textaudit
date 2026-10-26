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

Results below are from a `textaudit run --quick` run (4000 documents, 2 seeds). All numbers are on held-out test data (mean ± std over seeds).

### Preprocessing ablation

Keeping headers, footers and quoted text (`keep_metadata`) produces the largest change by far: +0.12 to +0.18 accuracy relative to a cleaned baseline. Other factors (lowercasing, stop-word removal, `min_df`, bigrams) change accuracy by only a few points. Complement Naive Bayes is slightly stronger than linear SVM on the cleaned data; the ranking reverses once metadata is kept.

| config | model | accuracy | macro-F1 | Δ accuracy vs base |
|---|---|---|---|---|
| base | complement_nb | 0.727 ± 0.033 | 0.704 ± 0.028 | — |
| base | linear_svm | 0.697 ± 0.034 | 0.683 ± 0.034 | — |
| keep_metadata | linear_svm | 0.875 ± 0.014 | 0.872 ± 0.017 | **+0.178** |
| keep_metadata | complement_nb | 0.847 ± 0.004 | 0.841 ± 0.005 | **+0.120** |
| min_df_1 | linear_svm | 0.714 ± 0.030 | 0.700 ± 0.027 | +0.017 |
| min_df_1 | complement_nb | 0.729 ± 0.042 | 0.708 ± 0.038 | +0.002 |
| bigrams | linear_svm | 0.699 ± 0.033 | 0.683 ± 0.028 | +0.003 |
| bigrams | complement_nb | 0.718 ± 0.030 | 0.694 ± 0.026 | −0.009 |
| stopwords | linear_svm | 0.705 ± 0.046 | 0.691 ± 0.045 | +0.008 |
| stopwords | complement_nb | 0.721 ± 0.035 | 0.698 ± 0.031 | −0.006 |
| no_lowercase | linear_svm | 0.691 ± 0.024 | 0.676 ± 0.021 | −0.006 |
| no_lowercase | complement_nb | 0.724 ± 0.023 | 0.701 ± 0.020 | −0.003 |
| min_df_5 | linear_svm | 0.676 ± 0.027 | 0.665 ± 0.028 | −0.021 |
| min_df_5 | complement_nb | 0.695 ± 0.034 | 0.673 ± 0.030 | −0.032 |

### Near-duplicate leakage

On the cleaned 20 Newsgroups subset, only 0.3–1.0 % of test documents have a near-duplicate (cosine ≥ 0.80–0.95) in the training set; those few examples are classified perfectly. With metadata kept the fraction rises slightly and the accuracy gap between flagged and unflagged documents is smaller, consistent with the overall higher accuracy.

| dataset | similarity ≥ | test docs flagged | accuracy on flagged | accuracy on unflagged | overall accuracy |
|---|---|---|---|---|---|
| 20ng-clean | 0.80 | 1.0 % | 100.0 % | 69.4 % | 69.7 % |
| 20ng-clean | 0.90 | 0.4 % | 100.0 % | 69.6 % | 69.7 % |
| 20ng-clean | 0.95 | 0.3 % | 100.0 % | 69.6 % | 69.7 % |
| 20ng-with-metadata | 0.80 | 2.2 % | 93.8 % | 87.3 % | 87.5 % |
| 20ng-with-metadata | 0.90 | 0.6 % | 83.3 % | 87.5 % | 87.5 % |
| 20ng-with-metadata | 0.95 | 0.4 % | 80.0 % | 87.6 % | 87.5 % |

### Label noise

Self-confidence (out-of-fold probability of the assigned label) detects injected symmetric noise well (AUROC 0.91–0.95). Removing the least-confident examples recovers a substantial fraction of the lost accuracy, though it does not fully restore the clean-label performance at higher noise rates.

| noise rate | detection AUROC | precision@k | test acc (noisy labels) | test acc (flagged removed) | test acc (true labels) |
|---|---|---|---|---|---|
| 0 % | n/a | n/a | 0.696 ± 0.027 | 0.696 ± 0.027 | 0.696 ± 0.027 |
| 5 % | 0.954 ± 0.001 | 0.547 ± 0.013 | 0.686 ± 0.029 | 0.694 ± 0.029 | 0.696 ± 0.027 |
| 10 % | 0.943 ± 0.002 | 0.627 ± 0.024 | 0.671 ± 0.042 | 0.686 ± 0.035 | 0.696 ± 0.027 |
| 20 % | 0.925 ± 0.001 | 0.685 ± 0.001 | 0.633 ± 0.023 | 0.659 ± 0.042 | 0.696 ± 0.027 |
| 30 % | 0.914 ± 0.007 | 0.741 ± 0.013 | 0.591 ± 0.024 | 0.650 ± 0.035 | 0.696 ± 0.027 |

**Take-away.** On this corpus the single largest uncontrolled factor is whether newsgroup headers and quoted text are left in the documents. Near-duplicate leakage is modest after basic cleaning. Simple self-confidence is already a useful signal for detecting random label noise.

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
