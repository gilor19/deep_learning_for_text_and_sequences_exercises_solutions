import argparse
import pickle
from typing import List
from utils import Sentence
from bilstm import BiLSTMModel, EMB_DIM, HIDDEN_DIM, LSTM_OUT_DIM
import torch


PAD = '<PAD>'
UNK = '<UNK>'


def create_sentences(vocab, file_path, representation) -> List[Sentence]:
    """
    Parse the data file and create a list of sentences.
    """
    sentences = []
    unk_lst = []
    if representation == 'a':
        sentences, unk_lst = create_word_sentences(vocab, file_path)
    elif representation == 'b':
        sentences, unk_lst = create_char_sentences(vocab, file_path)
    elif representation == 'c':
        sentences, unk_lst = create_pref_suff_sentences(vocab, file_path)
    elif representation == 'd':
        sentences, unk_lst = create_word_sentences(vocab, file_path)
        sentences_chars, _ = create_char_sentences(vocab, file_path)
        for i in range(len(sentences)):
            sentences[i].set_chars(sentences_chars[i].chars)
    return sentences, unk_lst


def create_word_sentences(vocab, file_path) -> List[Sentence]:
    sentences = []
    unk_test_map = []
    with open(file_path, "r") as f:
        words = []
        for line in f:
            if line == '\n':
                sentences.append(Sentence(words=torch.LongTensor(words), length=len(words)))
                count_sent_length(words)
                words = []
                continue
            text = line.strip()
            if text not in vocab.word2idx:
                unk_test_map.append(text)
                text = UNK
            # identify the longest word for padding
            count_char_length(text)
            words.append(vocab.word2idx[text])
    return sentences, unk_test_map


def create_char_sentences(vocab, file_path):
    sentences = []
    unk_map = []
    with open(file_path, "r") as f:
        word_chars = []
        for line in f:
            if line == '\n':
                sentences.append(Sentence(chars=word_chars, length=len(word_chars)))
                count_sent_length(word_chars)
                word_chars = []
                continue
            text = line.strip()
            count_char_length(text)
            for c in text:
                if c not in vocab.char2idx:
                    unk_map.append(c)
            word_chars.append(torch.tensor([vocab.char2idx[c] if c in vocab.char2idx
                                            else vocab.char2idx[UNK] for c in text], dtype=torch.long))
    return sentences, unk_map


def create_pref_suff_sentences(vocab, file_path):
    sentences = []
    unk_test_map = []
    with open(file_path, "r") as f:
        words, pref, suff = [], [], []
        for line in f:
            if line == '\n':
                sentences.append(Sentence(words=torch.LongTensor(words), length=len(words),
                                          prefs=torch.LongTensor(pref), suffs=torch.LongTensor(suff)))
                count_sent_length(words)
                words, pref, suff = [], [], []
                continue
            text = line.strip()
            pref_text = text[:3]
            suff_text = text[-3:]
            if text not in vocab.word2idx:
                unk_test_map.append(text)
                text = UNK
            if pref_text not in vocab.prefix2idx:
                pref_text = UNK
            if suff_text not in vocab.suffix2idx:
                suff_text = UNK
            count_char_length(text)
            words.append(vocab.word2idx[text])
            pref.append(vocab.prefix2idx[pref_text])
            suff.append(vocab.suffix2idx[suff_text])
    return sentences, unk_test_map


def count_char_length(text):
    if len(list(text)) > Sentence.max_length_char:
        Sentence.max_length_char = len(list(text))


def count_sent_length(words):
    if len(words) > Sentence.max_length_sent:
        Sentence.max_length_sent = len(words)


def predict_from_loaded_model(model_path, data, vocab, representation):
    model = BiLSTMModel(emb_dim=EMB_DIM, lstm_out_dim=LSTM_OUT_DIM, target_size=len(vocab.label2idx),
                        vocab=vocab, hidden_dim=HIDDEN_DIM, representation=representation)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()
    with torch.no_grad():
        for sent in data:
            output = model(sent, representation)
            _, pred = torch.max(output, 1)
            sent.label = pred
    return data


def write_pred_to_file(output_data, save_path, vocab, unk_lst, representation):
    with open(save_path, "w") as f:
        for sent in output_data:
            if representation in ['a', 'c', 'd']:
                for i, word_idx in enumerate(sent.words):
                    word = vocab.idx2word[word_idx.item()]
                    if word == UNK:
                        word = unk_lst.pop(0)
                    f.write(word+'\t'+vocab.idx2label[sent.label[i].item()]+'\n')
            elif representation == 'b':
                for i, chars in enumerate(sent.chars):
                    word = ''
                    for c in chars:
                        c = vocab.idx2char[c.item()]
                        if c == UNK:
                            c = unk_lst.pop(0)
                        word += c
                    f.write(word+'\t'+vocab.idx2label[sent.label[i].item()]+'\n')
            f.write('\n')


def main(representation, input_file, model_file, vocab_file, pred_path):
    file = open(vocab_file, 'rb')
    vocab = pickle.load(file)
    file.close()
    input_data, unk_list = create_sentences(vocab, input_file, representation)
    output = predict_from_loaded_model(model_path=model_file, data=input_data, vocab=vocab, representation=representation)
    write_pred_to_file(output, pred_path, vocab, unk_list, representation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repr", type=str, choices=['a', 'b', 'c', 'd'], help="representation")
    parser.add_argument("inputFile", type=str, help="path to input file")
    parser.add_argument("modelFile", type=str, help="path to model file")
    parser.add_argument("vocabFile", type=str, help="path to vocabulary file")
    parser.add_argument("predFile", type=str, help="path to saved prediction file")
    args = parser.parse_args()
    main(args.repr, args.inputFile, args.modelFile, args.vocabFile, args.predFile)
