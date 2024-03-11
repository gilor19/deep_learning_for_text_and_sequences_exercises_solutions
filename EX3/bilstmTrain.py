import argparse
import pickle
from collections import Counter
from typing import List
import torch
from torch import nn
from utils import Vocab, Sentence
from bilstm import BiLSTMModel, EMB_DIM, HIDDEN_DIM, LSTM_OUT_DIM


PAD = '<PAD>'
UNK = '<UNK>'


def create_word_sentences(vocab, file_path) -> List[Sentence]:
    """
    Parse the data file by words, each sentence will contain a representation by the sequence of its words.
    """
    sentences = []
    with open(file_path, "r") as f:
        words = []
        labels = []
        for line in f:
            if line == '\n':
                sentences.append(Sentence(words=torch.LongTensor(words), label=torch.LongTensor(labels),
                                          length=len(words)))
                count_sent_length(words)   # for padding logic
                words, labels = [], []
                continue
            text, label = line.strip().split()
            if text not in vocab.word2idx:
                text = UNK

            # identify the longest word for padding
            count_char_length(text)
            words.append(vocab.word2idx[text])
            labels.append(vocab.label2idx[label])
    return sentences


def count_char_length(text):  # for padding logic
    if len(list(text)) > Sentence.max_length_char:
        Sentence.max_length_char = len(list(text))


def count_sent_length(words):  # for padding logic
    if len(words) > Sentence.max_length_sent:
        Sentence.max_length_sent = len(words)


def create_char_sentences(vocab, file_path) -> List[Sentence]:
    """
    Parse the data file by characters, each sentence will contain a representation by the sequences of each word's
    characters.
    """
    sentences = []
    with open(file_path, "r") as f:
        word_chars = []
        labels = []
        for line in f:
            if line == '\n':
                sentences.append(Sentence(chars=word_chars, label=torch.LongTensor(labels),
                                          length=len(word_chars)))
                count_sent_length(word_chars)
                word_chars, labels = [], []
                continue
            text, label = line.strip().split()
            count_char_length(text)
            word_chars.append(torch.tensor([vocab.char2idx[c] if c in vocab.char2idx
                                            else vocab.char2idx[UNK] for c in text], dtype=torch.long))
            labels.append(vocab.label2idx[label])
    return sentences


def create_pref_suff_sentences(vocab, file_path) -> List[Sentence]:
    """
    Parse the data file by prefixes and suffixes, each sentence will contain a representation by the sequences
    of each word's prefix and suffix
    """
    sentences = []
    with open(file_path, "r") as f:
        words, pref, suff, labels = [], [], [], []
        for line in f:
            if line == '\n':
                sentences.append(Sentence(words=torch.LongTensor(words), label=torch.LongTensor(labels),
                                          length=len(words), prefs=torch.LongTensor(pref), suffs=torch.LongTensor(suff)))
                count_sent_length(words)
                words, pref, suff, labels = [], [], [], []
                continue
            text, label = line.strip().split()
            pref_text = text[:3]
            suff_text = text[-3:]
            if text not in vocab.word2idx:
                text = UNK
            if pref_text not in vocab.prefix2idx:
                pref_text = UNK
            if suff_text not in vocab.suffix2idx:
                suff_text = UNK
            count_char_length(text)
            words.append(vocab.word2idx[text])
            pref.append(vocab.prefix2idx[pref_text])
            suff.append(vocab.suffix2idx[suff_text])
            labels.append(vocab.label2idx[label])
    return sentences


def create_sentences(vocab, file_path, representation) -> List[Sentence]:
    """
    Parse the data file and create a list of sentences.
    """
    sentences = []
    if representation == 'a':
        sentences = create_word_sentences(vocab, file_path)
    elif representation == 'b':
        sentences = create_char_sentences(vocab, file_path)
    elif representation == 'c':
        sentences = create_pref_suff_sentences(vocab, file_path)
    elif representation == 'd':
        sentences = create_word_sentences(vocab, file_path)
        sentences_chars = create_char_sentences(vocab, file_path)
        for i in range(len(sentences)):
            sentences[i].set_chars(sentences_chars[i].chars)
    return sentences


def create_words_vocab(train_file, label_counter, labels2idx):
    words_counter = Counter()
    with open(train_file, "r") as f:
        for line in f:
            if line == '\n':
                continue
            text, label = line.strip().split()
            if label not in labels2idx:
                labels2idx[label] = label_counter
                label_counter += 1
            words_counter.update([text])
    # delete words that appear less than 2 times and delete the first 500 such words from the counter
    least_common = words_counter.most_common()[:-500:-1]
    for word, count in least_common:
        del words_counter[word]
    # add unk to the vocab
    words2idx = {PAD: 0, UNK: 1}
    words2idx.update(
        {word: idx + 2 for idx, word in enumerate(words_counter.keys())})
    idx2words = {idx: word for idx, word in enumerate(words2idx.keys())}
    idx2labels = {idx: label for label, idx in labels2idx.items()}
    return words2idx, idx2words, labels2idx, idx2labels


def create_pref_suff_vocab(vocab):
    prefix_set = set([word[:3] for word in vocab])
    prefix2idx = {UNK: 0}
    prefix2idx.update(
        {p: idx + 1 for idx, p in enumerate(list(prefix_set))})
    suffix_set = set([word[-3:] for word in vocab])
    suffix2_idx = {UNK: 0}
    suffix2_idx.update(
        {s: idx + 1 for idx, s in enumerate(list(suffix_set))})
    idx2prefix = {idx: pref for pref, idx in prefix2idx.items()}
    idx2suffix = {idx: suff for suff, idx in suffix2_idx.items()}
    return prefix2idx, suffix2_idx, idx2prefix, idx2suffix


def create_char_vocab(train_file, label_counter, labels2idx):
    char2idx = {str(i): i for i in range(10)}
    char2idx[UNK] = 10
    char_counter = 11
    with open(train_file, "r") as f:
        for line in f:
            if line == '\n':
                continue
            text, label = line.strip().split()
            if label not in labels2idx:
                labels2idx[label] = label_counter
                label_counter += 1
            for char in text:
                if char not in char2idx:
                    char2idx[char] = char_counter
                    char_counter += 1

    idx2char = {idx: char for char, idx in char2idx.items()}
    idx2labels = {idx: label for label, idx in labels2idx.items()}
    return char2idx, idx2char, labels2idx, idx2labels


def create_vocab(train_file):
    """
    Read the data and create words, prefixes, suffixes and characters Vocab object.
    """
    label_counter = 1
    labels2idx = {PAD: 0}
    words2idx, idx2words, labels2idx, idx2labels = create_words_vocab(train_file, label_counter, labels2idx)
    char2idx, idx2char, labels2idx, idx2labels = create_char_vocab(train_file, label_counter, labels2idx)
    prefix2idx, suffix2_idx, idx2prefix, idx2suffix = create_pref_suff_vocab(list(words2idx.keys()))
    vocab = Vocab(word2idx=words2idx, idx2word=idx2words, labels2idx=labels2idx,
                  prefix2idx=prefix2idx, suffix2idx=suffix2_idx, idx2label=idx2labels,
                  idx2prefix=idx2prefix, idx2suffix=idx2suffix,char2idx=char2idx, idx2char=idx2char)
    return vocab


def input_to_cuda(device, sentences):
    for sent in sentences:
        if sent.has_prefs():
            sent.prefs = sent.prefs.to(device)
            sent.suffs = sent.suffs.to(device)
        if sent.has_words():
            sent.words = sent.words.to(device)
        if sent.has_chars():
            temp = []
            for c in sent.chars:
                c = c.to(device)
                temp.append(c)
            sent.chars = temp
        sent.label = sent.label.to(device)


def main(representation, train_path_, dev_path_, is_ner_, model_path_, vocab_path_):
    vocab = create_vocab(train_path_)
    sentences = create_sentences(vocab, train_path_, representation)
    dev_sentences = create_sentences(vocab, dev_path_, representation)

    model = BiLSTMModel(emb_dim=EMB_DIM, lstm_out_dim=LSTM_OUT_DIM, target_size=len(vocab.label2idx),
                        vocab=vocab, hidden_dim=HIDDEN_DIM, representation=representation)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_to_cuda(device, sentences)
    input_to_cuda(device, dev_sentences)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    accuracies = []

    epochs = 5
    for epoch in range(epochs):
        model.train()
        cum_loss = 0
        for i, sent in enumerate(sentences):
            if i % 500 == 0:
                evaluate_model(criterion, dev_sentences, is_ner_, model, representation, vocab, accuracies)
            optimizer.zero_grad()
            outputs = model(sent, representation)
            loss = criterion(outputs, sent.label)
            cum_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        print(f"\nEpoch [{epoch+1}/{epochs}]:\nTrain Loss: {cum_loss/len(sentences):.4f}")

    torch.save(model.state_dict(), model_path_)
    pickle_file(f"{vocab_path_}.pkl", vocab)
    print("Model saved")
    print("##################################################")


def pickle_file(file_path, data):
    # open a file, where you want to store the data
    file = open(file_path, 'wb')

    # dump information to that file
    pickle.dump(data, file)

    # close the file
    file.close()


def evaluate_model(criterion, dev_sentences, is_ner_, model, representation, vocab, accuracies):
    model.eval()
    correct = 0
    os = 0
    dev_cum_loss = 0
    total = 0
    with torch.no_grad():
        for dev_sent in dev_sentences:
            outputs = model(dev_sent, representation)
            loss = criterion(outputs, dev_sent.label)
            dev_cum_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            if is_ner_:
                for i, p in enumerate(pred):
                    if p == vocab.label2idx['O'] == dev_sent.label[i]:
                        os += 1
                    else:
                        if p == dev_sent.label[i]:
                            correct += 1
            else:
                correct += pred.eq(dev_sent.label).sum().item()
            total += len(pred)

        accuracies.append(correct / (total - os))

        print(f"Dev Loss: {dev_cum_loss / len(dev_sentences):.4f}")
        print(f"Accuracy: {correct / (total - os)}")
        print("##################################################")
    model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repr", type=str, choices=['a', 'b', 'c', 'd'], help="representation")
    parser.add_argument("trainFile", type=str, help="path to train file")
    parser.add_argument("modelFile", type=str, help="path to save model file")
    parser.add_argument("vocabFile", type=str, help="path to save vocabulary file")
    parser.add_argument("--ner", help="is ner or not", action='store_true')
    parser.add_argument("--dev", type=str, help="path to dev file")
    args = parser.parse_args()
    main(args.repr, args.trainFile, args.dev, args.ner, args.modelFile, args.vocabFile)

