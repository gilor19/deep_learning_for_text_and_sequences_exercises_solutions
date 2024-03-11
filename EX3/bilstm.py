import torch
from torch import nn

EMB_DIM = 64
LSTM_OUT_DIM = 64
HIDDEN_DIM = 64


class BiLSTMModel(nn.Module):
    def __init__(self, emb_dim, lstm_out_dim, target_size, vocab, hidden_dim, representation):
        super(BiLSTMModel, self).__init__()

        if representation == 'a':
            self.word_embedding = nn.Embedding(len(vocab.word2idx), emb_dim)
        if representation == 'b':
            self.char_embedding = nn.Embedding(len(vocab.char2idx), emb_dim)
            self.lstm_chars = nn.LSTM(input_size=emb_dim, hidden_size=emb_dim)
        if representation == 'c':
            self.word_embedding = nn.Embedding(len(vocab.word2idx), emb_dim)
            self.pref_embedding = nn.Embedding(len(vocab.prefix2idx), emb_dim)
            self.suff_embedding = nn.Embedding(len(vocab.suffix2idx), emb_dim)
        if representation == 'd':
            self.word_embedding = nn.Embedding(len(vocab.word2idx), emb_dim)
            self.char_embedding = nn.Embedding(len(vocab.char2idx), emb_dim)
            self.lstm_chars = nn.LSTM(input_size=emb_dim, hidden_size=emb_dim)
            emb_dim = emb_dim * 2

        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=lstm_out_dim, bidirectional=True, num_layers=2)
        self.fc1 = nn.Linear(lstm_out_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, target_size)

    def forward(self, sent, representation):
        if representation == 'a':
            embeddings = self.handle_a_forward(sent)
        elif representation == 'b':
            embeddings = self.handle_b_forward(sent)
        elif representation == 'c':
            words = sent.words
            prefs = sent.prefs
            suffs = sent.suffs
            words_embeddings = self.word_embedding(words)
            prefs_embeddings = self.pref_embedding(prefs)
            suffs_embeddings = self.suff_embedding(suffs)
            embeddings = words_embeddings + prefs_embeddings + suffs_embeddings
        else:  # representation == 'd'
            word_emb = self.handle_a_forward(sent)
            char_emd = self.handle_b_forward(sent)
            embeddings = torch.cat((word_emb, char_emd.view(sent.length, -1)), dim=1)
        lstm_output, _ = self.lstm(embeddings.view(sent.length, 1, -1))
        out = self.fc1(lstm_output.view(sent.length, -1))
        out = torch.tanh(out)
        out = self.fc2(out)
        return out

    def handle_a_forward(self, sent):
        x = sent.words
        embeddings = self.word_embedding(x)
        return embeddings

    def handle_b_forward(self, sent):
        word_chars = sent.chars
        embeddings = []
        for word in word_chars:
            chars_embeddings = self.char_embedding(word)
            _, (hn, cn) = self.lstm_chars(chars_embeddings.view(len(word), 1, -1))
            embeddings.append(hn)
        embeddings = torch.cat(embeddings)
        return embeddings
