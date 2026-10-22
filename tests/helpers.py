import numpy as np

from textaudit.data import TextDataset


def synthetic_dataset(n_per_class=80, n_classes=4, seed=0, n_duplicates=0, leak_token=False):
    """Class-specific vocabulary mixed with shared words; optionally leaks the label as a token."""
    rng = np.random.default_rng(seed)
    shared = [f"common{i}" for i in range(60)]
    texts, labels = [], []
    for c in range(n_classes):
        own = [f"topic{c}word{i}" for i in range(25)]
        for _ in range(n_per_class):
            words = list(rng.choice(own, size=12)) + list(rng.choice(shared, size=18))
            if leak_token:
                words.append(f"leak{c}")
            rng.shuffle(words)
            texts.append(" ".join(words))
            labels.append(c)
    for i in range(n_duplicates):  # exact copies of existing documents
        texts.append(texts[i])
        labels.append(labels[i])
    return TextDataset("synthetic", texts, np.asarray(labels), [f"c{c}" for c in range(n_classes)])
