

class Sentence:
    max_length_sent = -1   # for padding logic
    max_length_char = -1   # for padding logic

    def __init__(self, words=None, chars=None, prefs=None,
                 suffs=None, length=None, label=None):
        self.words = words
        self.chars = chars
        self.prefs = prefs
        self.suffs = suffs
        self.label = label
        self.length = length

    def set_suffs(self, suffs):
        self.suffs = suffs

    def set_prefs(self, prefs):
        self.prefs = prefs

    def set_chars(self, chars):
        self.chars = chars

    def has_words(self):
        return self.words is not None

    def has_chars(self):
        return self.chars is not None

    def has_prefs(self):
        return self.prefs is not None

    def has_suffs(self):
        return self.suffs is not None

    def has_label(self):
        return self.label is not None

    def has_length(self):
        return self.length is not None


class Vocab:
    def __init__(self, word2idx=None, idx2word=None, labels2idx=None,
                 idx2label=None, char2idx=None, idx2char=None,
                 prefix2idx=None, idx2prefix=None, suffix2idx=None,
                 idx2suffix=None):
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.label2idx = labels2idx
        self.idx2label = idx2label
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.prefix2idx = prefix2idx
        self.idx2prefix = idx2prefix
        self.suffix2idx = suffix2idx
        self.idx2suffix = idx2suffix

    def has_word2idx(self):
        return self.word2idx is not None

    def has_idx2word(self):
        return self.idx2word is not None

    def has_label2idx(self):
        return self.label2idx is not None

    def has_idx2label(self):
        return self.idx2label is not None

    def has_char2idx(self):
        return self.char2idx is not None

    def has_idx2char(self):
        return self.idx2char is not None

    def has_prefix2idx(self):
        return self.prefix2idx is not None

    def has_idx2prefix(self):
        return self.idx2prefix is not None

    def has_suffix2idx(self):
        return self.suffix2idx is not None

    def has_idx2suffix(self):
        return self.idx2suffix is not None



