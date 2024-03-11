import numpy as np
import torch.nn as nn
import torch
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from argparse import ArgumentParser
import torch.autograd as autograd
from collections import Counter
import pickle


NOT_PRETRAINED = 'not_pretrained'
PRETRAINED = 'pretrained'
VECTORS = 'vectors'
HIDDEN_DIM = 'hidden_dim'
LABELS_DIM = 'labels_dim'
EMBED_DIM = 'embedding_dim'
VOCAB_SIZE = 'vocab_size'
NUM_CHARS = 'num_chars'
NUM_FILTERS = 'num_filters'
KERNEL_SIZE = 'kernel_size'
unk = 'UUUNKKK'
WINDOW_SIZE = 5
START_TOKEN = "<s>"
END_TOKEN = "</s>"
patterns = ['unk_all_capital', 'unk_first_capital', 'unk_ing', 'unk_tion', 'unk_all_digits']


class Window:
    def __init__(self, words, chars, label=None):
        self.words = words
        self.chars = chars
        self.label = label


class WindowModel(nn.Module):
    def __init__(self, network_args, load):
        super(WindowModel, self).__init__()
        if load:
            self.pretrained_vecs = torch.FloatTensor(network_args[PRETRAINED][VECTORS])
            self.embeddings = torch.nn.Embedding.from_pretrained(self.pretrained_vecs, freeze=False)
            self.char_embedding = nn.Embedding(network_args[PRETRAINED][NUM_CHARS],
                                               network_args[PRETRAINED][EMBED_DIM])
            self.conv1d = nn.Conv1d(network_args[PRETRAINED][EMBED_DIM], network_args[PRETRAINED][NUM_FILTERS],
                                   kernel_size=network_args[PRETRAINED][KERNEL_SIZE])
            self.conv1d = nn.Conv1d(network_args[PRETRAINED][EMBED_DIM], network_args[PRETRAINED][NUM_FILTERS],
                                   kernel_size=network_args[PRETRAINED][KERNEL_SIZE])
            hidden_dim = network_args[PRETRAINED][HIDDEN_DIM]
            labels_dim = network_args[PRETRAINED][LABELS_DIM]

        else:
            self.embeddings = nn.Embedding(network_args[NOT_PRETRAINED][VOCAB_SIZE],
                                           network_args[NOT_PRETRAINED][EMBED_DIM])
            hidden_dim = network_args[NOT_PRETRAINED][HIDDEN_DIM]
            labels_dim = network_args[NOT_PRETRAINED][LABELS_DIM]

        self.linear1 = nn.Linear(self.conv1d.out_channels + 250, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, labels_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x_words, x_chars = x[0], x[1]

        # word embedding
        x_words = self.embeddings(x_words)
        x_words = x_words.view(-1, self.embeddings.embedding_dim * WINDOW_SIZE)

        # char embedding
        x_chars = self.char_embedding(x_chars)
        if len(x_chars.shape) != 3:
            x_chars = x_chars.view(1,x_chars.shape[0],x_chars.shape[1])
        x_chars = x_chars.view(x_chars.shape[0], x_chars.shape[2], x_chars.shape[1])
        conv_out = self.conv1d(x_chars)
        pooled_out, _ = torch.max(conv_out, dim=-1)

        # concat words and chars
        x = torch.cat((x_words, pooled_out), -1)

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


def create_sentences(fname, labels, vocab, load):
    sentences = []
    with open(fname) as f:
        # buffer start of the sentence
        sentence = [(unk, 0), (unk, 0)]
        for line in f:
            if line != "\n":
                text, label = line.strip().split()
                if load:
                    text = text.lower()
                sentence.append((text, labels[label]))
            else:
                # buffer end of the sentence
                sentence.extend([(unk, 0), (unk, 0)])
                sentences.append(sentence)
                sentence = [(unk, 0), (unk, 0)]
    return sentences


def create_windows(sent, max_length, vocab, char2i, test=False):
    windows = []
    padding_size = 2
    for sentence in sent:
        if not test:
            words = [w for w, l in sentence]
            labels = [l for w, l in sentence]
        else:
            words = [w for w in sentence]
        for i in range(2, len(sentence) - 2):
            window_words = words[i-2: i+3]
            window_chars = []
            for word in window_words:
                if len(word) > max_length - padding_size:
                    word = word[:max_length - padding_size]
                chars = list(word)
                pad = max_length - len(chars) - padding_size
                chars_idxs = [0]*pad + [char2i[char] for char in chars] + [0]*padding_size
                window_chars.extend(chars_idxs)
            window_word_idxs = []
            for word in window_words:
                if word not in vocab:
                    if load:
                        word = unk
                    else:
                        word = identify_pattern(word)
                window_word_idxs.append(vocab[word])
            if not test:
                windows.append(Window(window_word_idxs, window_chars, labels[i]))
            else:
                windows.append(Window(window_word_idxs, window_chars))
    return windows


def train(data_loader, model, opt, criterion):
    cum_loss = 0
    for batch_idx, (words, chars, labels) in enumerate(data_loader):
        words, chars, labels = autograd.Variable(words), autograd.Variable(chars), autograd.Variable(labels)
        opt.zero_grad()
        output = model((words, chars))
        loss = criterion(output, labels)
        loss.backward()
        opt.step()
        cum_loss += loss.item()
    return cum_loss/len(data_loader)


def prepare_data_loader(windows, batch_size=128):
    window_words, window_chars, window_labels = [], [], []
    for window in windows:
        window_words.append(window.words)
        window_chars.append(window.chars)
        window_labels.append(window.label)
    train_dataset = TensorDataset(torch.LongTensor(np.array(window_words)), torch.LongTensor(np.array(window_chars)), torch.LongTensor(np.array(window_labels)))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_data_loader


def plot_loss_acc(accuracies, dev_loss_list, epochs, train_loss_list, pos, ):

    epoch_list = np.arange(1, epochs + 1, 1)
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epoch_list, train_loss_list, label='train loss')
    plt.plot(epoch_list, dev_loss_list, label='dev loss')
    plt.legend()
    plt.savefig("loss.png")
    plt.figure()
    plt.title('Model Accuracy')
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


def create_test_sentences(fname, vocab, load):
    sentences = []
    unk_test_map = {}
    sentence_ends = []
    original_words = {}
    with open(fname) as f:
        # buffer start of the sentence
        sentence = [unk, unk]
        for idx, line in enumerate(f):
            if line != "\n":
                text = line.strip()
                original_words[idx] = text
                if load:
                    text = text.lower()
                sentence.append(text)
            else:
                # buffer end of the sentence
                sentence.extend([unk, unk])
                sentences.append(sentence)
                sentence = [unk, unk]
                sentence_ends.append(idx)
    return sentences, unk_test_map, sentence_ends, original_words


def preprocess(data, load):
    label2idx, word2idx = create_vocab(data)
    idx_2_word = {v: k for k, v in word2idx.items()}
    idx_2_label = {v: k for k, v in label2idx.items()}
    sentences = create_sentences(data, label2idx, word2idx, load)
    return sentences, idx_2_word, idx_2_label, word2idx, label2idx


def create_vocab(fname: str):
    """
    Create a vocabulary of the words and labels in the training set.

    Returns:
        words2idx
        labels2idx
    """
    words_counter = Counter()
    label_counter = 1
    labels2idx = {'None': 0}
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
        if pattern != unk:
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
        return unk


def handle_chars(words):
    unique_chars = set()
    max_length = -1
    for word in words:
        if len(word)> max_length:
            max_length = len(word)
        for char in word:
            unique_chars.add(char)

    chars2i = {k: idx+1 for idx, k in enumerate(unique_chars)}
    chars2i['pad'] = 0
    return chars2i, max_length


def main(train_path_, dev_path_, is_ner, test_path_=False, load_=False, lr=0.0001, epochs=6):
    pos = train_path_.split('/')[-2] == 'pos'
    task = "POS" if pos else "NER"
    network_args = {}
    hidden_dim = 64
    embedding_dim = 50
    num_filters = 30
    kernel_size = 3
    if load_:
        vectors = np.loadtxt("wordVectors.txt")
        with open("vocab.txt") as file:
            vocab = {line.rstrip(): idx for idx, line in enumerate(file)}
        i2w = {v: k for k, v in vocab.items()}
        c2i, longest_word_length = handle_chars(vocab)
        print("Collecting labels")
        labels = collect_labels(train_path_)
        i2l = {v: k for k, v in labels.items()}
        train_sentences = create_sentences(train_path_, labels, vocab, load_)
        network_args[PRETRAINED] = {VECTORS: vectors, HIDDEN_DIM: hidden_dim, LABELS_DIM: len(i2l), NUM_CHARS: len(c2i),
                                    NUM_FILTERS: num_filters, KERNEL_SIZE: kernel_size, EMBED_DIM: embedding_dim}
    else:
        print("Creating vocabulary")
        train_sentences, i2w, i2l, vocab, labels = preprocess(train_path_, load_)
        network_args[NOT_PRETRAINED] = {VOCAB_SIZE: len(i2w), EMBED_DIM: embedding_dim,
                                        LABELS_DIM: len(i2l), HIDDEN_DIM: hidden_dim}
    print("Preparing train data")
    train_windows = create_windows(train_sentences, longest_word_length, vocab, c2i)
    model = WindowModel(network_args, load_)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    data_loader = prepare_data_loader(train_windows)

    print("Preparing dev data")
    dev_sentences = create_sentences(dev_path_, labels, vocab, load_)
    dev_windows = create_windows(dev_sentences, longest_word_length, vocab, c2i)
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
        with torch.no_grad():
            for batch_idx, (words, chars, label) in enumerate(dev_data_loader):
                output = model((words, chars))
                pred = output.argmax()
                dev_loss = criterion(output, label)
                cum_loss += dev_loss.item()
                if is_ner:
                    if pred.item() == label.item() == labels['O']:
                        os += 1
                        continue
                correct += pred.eq(label).sum().item()
        print(f"dev loss: {cum_loss / (len(dev_windows))}")
        print(f"Accuracy: {correct / (len(dev_windows) - os)}")
        accuracies.append(correct / (len(dev_windows) - os))
        dev_loss_list.append(cum_loss / len(dev_data_loader))

    plot_loss_acc(accuracies, dev_loss_list, epochs, train_loss_list, pos)

    # save files for filter analysis
    filters = model.conv1d.weight
    pickle.dump(filters, open(f'{task}_filters.dat', 'wb'))
    chars_embedding = model.char_embedding.weight
    pickle.dump(chars_embedding, open(f'{task}_char_embedding.dat', 'wb'))
    pickle.dump(c2i, open(f'{task}_char2idx.dat', 'wb'))

    if test_path_: 
        test_sent, unk_test_map, sentences_end, original_words = create_test_sentences(test_path_, vocab, load_)
        test_windows = create_windows(test_sent,longest_word_length, vocab, c2i, test=True)

        res = []
        for window in test_windows:
            y_hat = model((torch.LongTensor(window.words), torch.LongTensor(window.chars))).argmax().item()
            res.append((i2w[window.words[2]], i2l[y_hat]))

        write_test(pos, res, sentences_end, original_words)


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
