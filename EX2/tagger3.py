import numpy as np
import torch.nn as nn
import torch
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from argparse import ArgumentParser
import torch.autograd as autograd
from collections import Counter

NOT_PRETRAINED = 'not_pretrained'
PRETRAINED = 'pretrained'
VECTORS = 'vectors'
HIDDEN_DIM = 'hidden_dim'
LABELS_DIM = 'labels_dim'
EMBED_DIM = 'embedding_dim'
VOCAB_SIZE = 'vocab_size'
unk = 'UUUNKKK'
WINDOW_SIZE = 5
START_TOKEN = "<s>"
END_TOKEN = "</s>"
patterns = ['unk_all_capital', 'unk_first_capital', 'unk_ing', 'unk_tion', 'unk_all_digits']


class Window:
    def __init__(self, words, prefixes=None, suffixes=None, label=None):
        self.words = words
        self.label = label
        self.prefixes = prefixes
        self.suffixes = suffixes


class WindowModel(nn.Module):
    def __init__(self, network_args, pref, suff):
        super(WindowModel, self).__init__()
        if load:
            self.pretrained_vecs = torch.FloatTensor(network_args[PRETRAINED][VECTORS])
            self.embeddings = torch.nn.Embedding.from_pretrained(self.pretrained_vecs, freeze=False)
            hidden_dim = network_args[PRETRAINED][HIDDEN_DIM]
            labels_dim = network_args[PRETRAINED][LABELS_DIM]
            embedding_dim = 50   # the size of the vectors received in the assignment definition
        else:
            self.embeddings = nn.Embedding(network_args[NOT_PRETRAINED][VOCAB_SIZE],
                                           network_args[NOT_PRETRAINED][EMBED_DIM])
            hidden_dim = network_args[NOT_PRETRAINED][HIDDEN_DIM]
            labels_dim = network_args[NOT_PRETRAINED][LABELS_DIM]
            embedding_dim = network_args[NOT_PRETRAINED][EMBED_DIM]

        self.embeddings_prefix = nn.Embedding(len(pref), embedding_dim)
        self.embeddings_suffix = nn.Embedding(len(suff), embedding_dim)
        self.linear1 = nn.Linear(self.embeddings.embedding_dim * WINDOW_SIZE, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, labels_dim)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        words = x[:, 0, :]
        prefixes = x[:, 1, :]
        suffixes = x[:, 2, :]
        word_vec = self.embeddings(words)
        pref_vec = self.embeddings_prefix(prefixes)
        suff_vec = self.embeddings_suffix(suffixes)
        sum_vec = word_vec + pref_vec + suff_vec
        # flatten the embeddings
        x = sum_vec.view(-1, self.embeddings.embedding_dim * WINDOW_SIZE)
        x = self.linear1(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


def collect_labels(fname):
    label_counter = 1
    labels2idx = {'None': 0}

    with open(fname) as f:
        for line in f:
            if line != "\n":
                text, label = line.strip().split()
                if label not in labels2idx:
                    labels2idx[label] = label_counter
                    label_counter += 1
    return labels2idx


def create_sentences(fname, labels, vocab, pref_vocab, suff_vocab, load):
    sentences = []
    pref_sentences = []
    suff_sentences = []
    with open(fname) as f:
        # buffer start of the sentence
        sentence = [(vocab[unk], 0), (vocab[unk], 0)]
        pref_sentence = [pref_vocab[unk], pref_vocab[unk]]
        suff_sentence = [suff_vocab[unk], suff_vocab[unk]]
        for line in f:
            if line != "\n":
                text, label = line.strip().split()
                if load:
                    text = text.lower()
                pref, suff = text[:3], text[-3:]
                if text not in vocab:  # replace low count words with unknown
                    text = unk
                if pref not in pref_vocab:
                    pref = unk
                if suff not in suff_vocab:
                    suff = unk
                sentence.append((vocab[text], labels[label]))
                pref_sentence.append(pref_vocab[pref])
                suff_sentence.append(suff_vocab[suff])
            else:
                # buffer end of the sentence
                sentence.extend([(vocab[unk], 0), (vocab[unk], 0)])
                sentences.append(sentence)
                pref_sentence.extend([pref_vocab[unk], pref_vocab[unk]])
                pref_sentences.append(pref_sentence)
                suff_sentence.extend([suff_vocab[unk], suff_vocab[unk]])
                suff_sentences.append(suff_sentence)

                sentence = [(vocab[unk], 0), (vocab[unk], 0)]
                pref_sentence = [pref_vocab[unk], pref_vocab[unk]]
                suff_sentence = [suff_vocab[unk], suff_vocab[unk]]
    return sentences, pref_sentences, suff_sentences


def create_windows(sent, pref, suff):
    windows = []
    for idx, sentence in enumerate(sent):
        words = [w for w, l in sentence]
        labels = [l for w, l in sentence]
        window_pref = [p for p in pref[idx]]
        window_suff = [s for s in suff[idx]]
        for i in range(2, len(sentence) - 2):
            windows.append(Window(words[i-2: i+3], prefixes=window_pref[i-2: i+3],
                                  suffixes=window_suff[i-2: i+3], label=labels[i]))
    return windows


def train(data_loader, model, opt, criterion):
    cum_loss = 0
    for batch_idx, (data, labels) in enumerate(data_loader):
        data, labels = autograd.Variable(data), autograd.Variable(labels)
        opt.zero_grad()
        output = model(data)
        loss = criterion(output, labels)
        loss.backward()
        opt.step()
        cum_loss += loss.item()
    return cum_loss/len(data_loader)


def prepare_data_loader(windows, batch_size=128):
    window_input, window_labels = [], []
    for window in windows:
        window_input.append((window.words, window.prefixes, window.suffixes))
        window_labels.append(window.label)
    train_dataset = TensorDataset(torch.LongTensor(np.array(window_input)), torch.LongTensor(np.array(window_labels)))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_data_loader


def plot_loss_acc(accuracies, dev_loss_list, epochs, train_loss_list, pos):
    epoch_list = np.arange(1, epochs + 1, 1)
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epoch_list, train_loss_list, label='train loss')
    plt.plot(epoch_list, dev_loss_list, label='dev loss')
    plt.legend()
    plt.savefig("loss.png")
    plt.figure()
    plt.title('Mdel Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epoch_list, accuracies, label='accuracy')
    plt.legend()
    plt.savefig("accuracy.png")


def write_test(pos, res, sentences_end, original_words):
    with open('test_preds', 'w') as f:
        idx = 0
        for word, label in res:
            if idx == sentences_end[0]:
                f.write('\n')
                idx += 1
                sentences_end.pop(0)
            word = original_words[idx]
            f.write(f"{word} {label}\n")
            idx += 1
        f.write('\n')


def create_test_sentences(fname, vocab, pref_vocab, suff_vocab, load):
    sentences = []
    pref_sentences = []
    suff_sentences = []
    unk_test_map = {}
    sentence_ends = []
    original_words = {}
    with open(fname) as f:
        # buffer start of the sentence
        sentence = [vocab[unk], vocab[unk]]
        pref_sentence = [pref_vocab[unk], pref_vocab[unk]]
        suff_sentence = [suff_vocab[unk], suff_vocab[unk]]
        for idx, line in enumerate(f):
            if line != "\n":
                text = line.strip()
                original_words[idx] = text
                if load:
                    text = text.lower()
                pref, suff = text[:3], text[-3:]
                if text not in vocab:  # replace low count words with unknown
                    unk_test_map[idx] = text
                    text = unk
                if pref not in pref_vocab:
                    pref = unk
                if suff not in suff_vocab:
                    suff = unk
                sentence.append(vocab[text])
                pref_sentence.append(pref_vocab[pref])
                suff_sentence.append(suff_vocab[suff])
            else:
                # buffer end of the sentence
                sentence.extend([vocab[unk], vocab[unk]])
                sentences.append(sentence)
                pref_sentence.extend([pref_vocab[unk], pref_vocab[unk]])
                pref_sentences.append(pref_sentence)
                suff_sentence.extend([suff_vocab[unk], suff_vocab[unk]])
                suff_sentences.append(suff_sentence)

                sentence = [vocab[unk], vocab[unk]]
                pref_sentence = [pref_vocab[unk], pref_vocab[unk]]
                suff_sentence = [suff_vocab[unk], suff_vocab[unk]]

                sentence_ends.append(idx)
    return sentences, pref_sentences, suff_sentences, unk_test_map, sentence_ends, original_words


def create_test_windows(sent, pref, suff):
    windows = []
    for idx, sentence in enumerate(sent):
        words = [w for w in sentence]
        window_pref = [p for p in pref[idx]]
        window_suff = [s for s in suff[idx]]
        for i in range(2, len(sentence) - 2):
            windows.append(Window(words[i-2: i+3], prefixes=window_pref[i-2: i+3],
                                  suffixes=window_suff[i-2: i+3]))
    return windows


def preprocess(data, load):
    label2idx, word2idx = create_vocab(data)
    idx_2_word = {v: k for k, v in word2idx.items()}
    idx_2_label = {v: k for k, v in label2idx.items()}
    prefix, suffix = create_pref_suff(word2idx)
    sentences, pref_sentences, suff_sentences = create_sentences(data, label2idx, word2idx, prefix, suffix, load)
    return sentences, pref_sentences, suff_sentences, idx_2_word, idx_2_label, word2idx, label2idx, prefix, suffix


def create_vocab(fname: str):
    """
    Create a vocabulary of the words and labels in the training set.

    Returns:
        words2idx
        labels2idx
    """
    words_counter = Counter()
    label_counter = 0
    labels2idx = {}
    with open(fname) as f:
        for line in f:
            if line != "\n":
                text, label = line.strip().split()
                if label not in labels2idx:
                    labels2idx[label] = label_counter
                    label_counter += 1
                words_counter.update([text])

    # collect unknown representatives
    unk_candidates = [k for k, v in words_counter.items() if v < 2]
    unk_representatives = [[] for pattern in patterns]
    for word in unk_candidates:
        pattern = identify_pattern(word)
        if pattern != 'unk':
            unk_representatives[patterns.index(pattern)].append(word)
    unk_representatives = [l[:300] for l in unk_representatives]

    # delete the unk representatives from the words counter
    for pattern in unk_representatives:
        for word in pattern:
            del words_counter[word]

    # create vocabulary
    words2idx = {word: idx for idx, word in enumerate(words_counter.keys())}
    words2idx[START_TOKEN] = len(words2idx)
    words2idx[END_TOKEN] = len(words2idx)
    words2idx[unk] = len(words2idx)
    for pattern in patterns:
        words2idx[pattern] = len(words2idx)

    return labels2idx, words2idx


def identify_pattern(word: str):
    if word.isupper():
        return 'unk_all_capital'
    elif word[0].isupper():
        return 'unk_first_capital'
    elif word[-3:] == 'ing':
        return 'unk_ing'
    elif word[-4:] == 'tion':
        return 'unk_tion'
    elif word.isdigit():
        return 'unk_all_digits'
    else:
        return 'unk'


def main(train_path_, dev_path_, is_ner, test_path_=False, load_=False, lr=0.0001, epochs=6):
    pos = train_path_.split('/')[-2] == 'pos'
    network_args = {}
    hidden_dim = 64
    embedding_dim = 50
    if load_:
        vectors = np.loadtxt("wordVectors.txt")
        with open("vocab.txt") as file:
            vocab = {line.rstrip(): idx for idx, line in enumerate(file)}
        i2w = {v: k for k, v in vocab.items()}
        print("Collecting labels")
        labels = collect_labels(train_path_)
        i2l = {v: k for k, v in labels.items()}
        prefix, suffix = create_pref_suff(vocab)
        train_sentences, pref_sentences, suff_sentences = create_sentences(train_path_, labels, vocab,
                                                                           prefix, suffix, load_)
        network_args[PRETRAINED] = {VECTORS: vectors, HIDDEN_DIM: hidden_dim, LABELS_DIM: len(i2l)}
    else:
        print("Creating vocabulary")
        train_sentences, pref_sentences, suff_sentences, i2w, i2l, vocab, labels, prefix, suffix =\
            preprocess(train_path_, load_)
        network_args[NOT_PRETRAINED] = {VOCAB_SIZE: len(i2w), EMBED_DIM: embedding_dim,
                                        LABELS_DIM: len(i2l), HIDDEN_DIM: hidden_dim}
    print("Preparing train data")

    train_windows = create_windows(train_sentences, pref_sentences, suff_sentences)
    model = WindowModel(network_args, prefix, suffix)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    epochs = epochs
    data_loader = prepare_data_loader(train_windows)

    print("Preparing dev data")
    dev_sentences, dev_pref, dev_suff = create_sentences(dev_path_, labels, vocab, prefix, suffix, load_)
    dev_windows = create_windows(dev_sentences, dev_pref, dev_suff)
    dev_data_loader = prepare_data_loader(dev_windows, batch_size=1)

    print("Training the model")
    accuracies = []
    dev_loss_list = []
    train_loss_list = []

    for epoch in range(epochs):
        model.train()
        train_loss = train(data_loader, model, optimizer, criterion)
        train_loss_list.append(train_loss)
        print(f"Epoch: {epoch}, train loss: {train_loss}")

        model.eval()
        correct = 0
        os = 0
        print("Starting evaluation")
        cum_loss = 0
        predictions = set()
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(dev_data_loader):
                output = model(data)
                pred = output.argmax()
                predictions.add(pred.item())
                dev_loss = criterion(output, label)
                cum_loss += dev_loss.item()
                if is_ner:
                    if pred.item() == label.item() == labels['O']:
                        os += 1
                        continue
                correct += pred.eq(label).sum().item()
        print(f"dev loss: {cum_loss / (len(dev_windows) - os)}")
        print(f"Accuracy: {correct / (len(dev_windows) - os)}")
        accuracies.append(correct / (len(dev_windows) - os))
        dev_loss_list.append(cum_loss / len(dev_data_loader))

    plot_loss_acc(accuracies, dev_loss_list, epochs, train_loss_list, pos)

    if test_path_:
        test_sent, pref_sent, suff_sent, unk_test_map, sentences_end, original_words = create_test_sentences(test_path_,
                                                                                       vocab, prefix, suffix, load_)
        test_windows = create_test_windows(test_sent, pref_sent, suff_sent)

        res = []
        for window in test_windows:
            input_tensor = torch.LongTensor((window.words, window.prefixes, window.suffixes)).resize_(1,3,5)
            y_hat = model(input_tensor).argmax().item()
            res.append((i2w[window.words[2]], i2l[y_hat]))

        write_test(pos, res, sentences_end, original_words)


def create_pref_suff(vocab):
    prefix_set = set([word[:3] for word in vocab])
    prefix = {p: idx for idx, p in enumerate(list(prefix_set))}
    prefix[unk] = len(prefix)
    suffix_set = set([word[-3:] for word in vocab])
    suffix = {s: idx for idx, s in enumerate(list(suffix_set))}
    suffix[unk] = len(suffix)
    return prefix, suffix


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-t", "--train", dest="train", help="path to train set")
    parser.add_argument("-d", "--dev", dest="dev", help="path to dev set")
    parser.add_argument("-l", "--load", dest="load", help="load pretrained vectors", action='store_true')
    parser.add_argument("--test", dest="test", help="path to test set")
    parser.add_argument("-lr", "--learning_rate", dest="learning_rate", help="learning rate size", type=float)
    parser.add_argument("-e", "--epochs", dest="epochs", help="number of epochs", type=int)
    parser.add_argument("-n", "--ner", dest="is_ner", help="ner flag to adjust accuracy calculation", action='store_true')
    args = parser.parse_args()
    train_path = args.train
    dev_path = args.dev
    test_path = args.test
    load = args.load
    learning_rate = args.learning_rate
    epochs = args.epochs
    is_ner = args.is_ner
    main(train_path, dev_path, is_ner, test_path, load, learning_rate, epochs)
