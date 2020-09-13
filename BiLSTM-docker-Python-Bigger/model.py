#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import necessary Python Packages
import torch
from torch import nn

device = torch.device("cpu")


def log_sum_exp(vec):
    """
    log(sum(exp(x))) Function
    """
    max_score = torch.max(vec, 0)[0].unsqueeze(0)
    max_score_broadcast = max_score.expand(vec.size(1), vec.size(1))
    result = max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), 0)).unsqueeze(0)
    return result.squeeze(1)


class BiLSTMCRF(nn.Module):
    """
    The BiLSTM-CRF with Attention Model
    """
    def __init__(
        self,
        tag_map={"O": 0, "START": 4, "STOP": 5},
        batch_size=256,
        vocab_size=20,
        hidden_dim=128,
        dropout=1.0,
        embedding_dim=100,
        max_length=200,
        start_tag="START",
        stop_tag="STOP"
    ):

        super(BiLSTMCRF, self).__init__()

        # Some hyper-parameters
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.tag_size = len(tag_map)
        self.tag_map = tag_map
        self.start_tag = start_tag
        self.stop_tag = stop_tag
        self.max_length = 200

        # Matrix of transition parameters. Entry i,j is the score of transitioning *to* i *from* j
        self.transitions = nn.Parameter(torch.randn(self.tag_size, self.tag_size, device=device))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[:, self.tag_map[self.start_tag]] = -1000.
        self.transitions.data[self.tag_map[self.stop_tag], :] = -1000.

        self.Dropout = nn.Dropout(self.dropout)
        # Get the word embeddings for the vocab (vocabulary)
        self.word_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.embedding_dim)

        # The model of BiLSTM
        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=self.hidden_dim // 2,
                            num_layers=1,
                            bidirectional=True,
                            batch_first=True,
                            dropout=self.dropout)

        self.linear1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        # This Linear Layer maps the output of the LSTM (hidden_dim) into tag space (tag_size).
        # The size of the output of this linear layer is tag_size
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)
        self.Leaky_ReLu = nn.LeakyReLU()

    def prediction(self, sentence, real_length):
        """
        Get model prediction
        :param sentence: Input sentence
        :return logits: Output predicted tags
        """
        # Initialize hidden layer
        self.length = sentence.shape[1]

        # Input character embedding
        embeddings = self.word_embeddings(sentence)
        embeddings = self.Dropout(embeddings)
        embeddings = embeddings.view(self.batch_size, self.length, self.embedding_dim)

        # BiLSTM Model
        packed_word = nn.utils.rnn.pack_padded_sequence(embeddings, real_length, batch_first=True, enforce_sorted=False)
        lstm_out, (_, _) = self.lstm(packed_word)
        unpacked_word, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, padding_value=0, total_length=self.max_length)

        # Linear Layer
        logits = self.Leaky_ReLu(self.linear1(unpacked_word))
        logits = self.Leaky_ReLu(self.linear2(logits))
        logits = self.Leaky_ReLu(self.hidden2tag(logits))
        return logits

    def neg_log_likelihood(self, sentences, tags, length):
        """
        Negative Log-Likelihood (NLL) Loss Function -> - (Real Path Score - Total Score)
        """
        self.batch_size = sentences.size(0)

        # Get the output tag_size tensor from the Linear Layer
        logits = self.prediction(sentence=sentences.to(device), real_length=length)
        real_path_score = torch.zeros(1, device=device)
        total_score = torch.zeros(1, device=device)

        for logit, tag, leng in zip(logits, tags, length):
            logit = logit[:leng].to(device)
            tag = tag[:leng].to(device)

            # Calculate the Real Path Score
            real_path_score += self.real_path_score(logit, tag).to(device)

            # Calculate the total score
            total_score += self.total_score(logit, tag).to(device)

        # Output the NLL Loss
        return total_score - real_path_score

    def forward(self, sentences, real_length, lengths=None):
        """
        Do the forward algorithm to compute the partition function
        """
        sentences = torch.tensor(sentences, dtype=torch.long, device=device)

        if not lengths:
            lengths = [i.size(-1) for i in sentences]

        self.batch_size = sentences.size(0)
        logits = self.prediction(sentence=sentences, real_length=real_length)

        scores = []
        paths = []
        for logit, leng in zip(logits, lengths):
            logit = logit[:leng]
            score, path = self.viterbi_decode(logit)
            scores.append(score)
            paths.append(path)
        return scores, paths

    def real_path_score(self, logits, label):
        """
        Calculate Real Path Score
        """
        score = torch.zeros(1, device=device)
        label = torch.cat([torch.tensor([self.tag_map[self.start_tag]], dtype=torch.long, device=device), label.to(torch.long)])
        for index, logit in enumerate(logits):
            emission_score = logit[label[index + 1]]
            transition_score = self.transitions[label[index], label[index + 1]]
            score += emission_score + transition_score

        # Add the final Stop Tag, the final transition score
        score += self.transitions[label[-1], self.tag_map[self.stop_tag]]
        return score

    def total_score(self, logits, label):
        """
        Calculate the total CRF Score
        """
        previous = torch.full((1, self.tag_size), 0, device=device)
        for index in range(len(logits)):
            previous = previous.expand(self.tag_size, self.tag_size).t()
            obs = logits[index].view(1, -1).expand(self.tag_size, self.tag_size)
            scores = previous + obs + self.transitions
            previous = log_sum_exp(scores)
        previous = previous + self.transitions[:, self.tag_map[self.stop_tag]]
        total_scores = log_sum_exp(previous.t())[0]
        return total_scores

    def viterbi_decode(self, logits):
        backpointers = []
        trellis = torch.zeros(logits.size(), device=device)
        backpointers = torch.zeros(logits.size(), dtype=torch.long, device=device)
        trellis[0] = logits[0]
        for t in range(1, len(logits)):
            v = trellis[t - 1].unsqueeze(1).expand_as(self.transitions) + self.transitions
            trellis[t] = logits[t] + torch.max(v, 0)[0]
            backpointers[t] = torch.max(v, 0)[1]
        viterbi = [torch.max(trellis[-1], -1)[1].cpu().tolist()]
        backpointers = backpointers.numpy()
        for bp in reversed(backpointers[1:]):
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()

        viterbi_score = torch.max(trellis[-1], 0)[0].tolist()
        return viterbi_score, viterbi
