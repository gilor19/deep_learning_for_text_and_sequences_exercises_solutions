from collections import Counter
from typing import Tuple, List


def write_preds(preds, fname):
    with open(fname, "w") as f:
        for p in preds:
            f.write(f"{p}\n")


class PreProcess:
    def __init__(self, train_path=None, dev_path=None, test_path=None, gram="bigram"):
        gram_method = self.text_to_bigrams
        if gram == "unigram":
            gram_method = self.text_to_unigrams

        self.train_data = [(lang, gram_method(text)) for lang, text in self.read_data(train_path)]
        self.dev_data = [(lang, gram_method(text)) for lang, text in self.read_data(dev_path)]
        if test_path:
            self.test_data = [gram_method(text) for _, text in self.read_data(test_path)]
        else:
            self.test_data = None

        self.counter = self.count_features()

        # 600 most common bigrams in the training set.
        self.vocab = set([x for x, c in self.counter.most_common(600)])

        # label strings to IDs
        self.L2I = {l: i for i, l in enumerate(list(sorted(set([l for l, t in self.train_data]))))}

        # ID to label strings
        self.I2L = {i: l for l, i in self.L2I.items()}

        # feature strings (bigrams) to IDs
        self.F2I = {f: i for i, f in enumerate(list(sorted(self.vocab)))}

    def count_features(self) -> Counter:
        counter = Counter()
        for l, feats in self.train_data:
            counter.update(feats)
        return counter

    def read_data(self, fname: str) -> List[Tuple[str, str]]:
        """
        Read data from a file.

        Args:
            fname: path to the file.

        Returns:
            A list of (label, text) pairs.
        """
        data = []
        with open(fname) as f:
            for line in f:
                label, text = line.strip().lower().split("\t", 1)
                data.append((label, text))
        return data

    def text_to_bigrams(self, text):
        return ["%s%s" % (c1, c2) for c1, c2 in zip(text, text[1:])]

    def text_to_unigrams(self, text):
        return [c for c in text]

