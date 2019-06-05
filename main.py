from __future__ import print_function

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable


def split_data(sentences, labels):
    len_feat = len(sentences)
    indexes = np.random.permutation(len_feat)
    sentences = np.array(sentences)[indexes]
    labels = np.array(labels)[indexes]
    split_frac = 0.8
    train_x = sentences[0:int(split_frac*len_feat)]
    train_y = labels[0:int(split_frac*len_feat)]
    test_x = sentences[int(split_frac*len_feat):]
    test_y = labels[int(split_frac*len_feat):]
    return train_x, train_y, test_x, test_y


def read_data(mlp=False):
    ## FAKE DATA
    dim_emb = 6 * 2 * 2
    dim_serie = 16
    n_embeddings = 50
    n_inputs = 1000
    embeddings = [ [ np.random.rand(dim_emb).astype(np.float32) for _ in range(dim_serie) ]
                   for _ in range(n_embeddings) ]
    labels = [ np.random.randint(n_embeddings) for _ in range(n_inputs) ]
    if mlp:
        input = [ np.array(embeddings[y]).flatten() for y in labels ]
    else:
        input = [ embeddings[y] for y in labels ]
    train_x, train_y, test_x, test_y = split_data(input, labels)
    return train_x, train_y, test_x, test_y


class RNN(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.rnn = nn.LSTM(params["input_dim"],
                           params["hidden_dim"],
                           num_layers=params["n_layers"],
                           bidirectional=params["bidirectional"],
                           dropout=params["dropout"])

        self.fc = nn.Linear(params["hidden_dim"], params["output_dim"])
        self.log_softmax = nn.LogSoftmax()
        self.dropout = nn.Dropout(params["dropout"])

    def forward(self, input):
        # input = [sent len, batch size]
        input = self.dropout(input)
        output, (h, c) = self.rnn(input)

        return self.log_softmax(self.fc(self.dropout(h)).squeeze(0))


class MLP(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.fc = nn.Linear(params["input_dim"], params["output_dim"])
        self.log_softmax = nn.LogSoftmax()

    def forward(self, input):
        return self.log_softmax(self.fc(input).squeeze(0))

def binary_accuracy(preds, y):
    _, preds = preds.max(1)
    correct = (preds == y).float()
    return correct.sum() / len(correct)


def train(model, params, inputs, labels, optimizer, criterion, grad=False):
    epoch_loss = 0
    epoch_acc = 0

    model.train() if grad else model.eval()
    batch_size = params["batch_size"]

    for batch in range(int(len(inputs) / batch_size)):
        if grad:
            optimizer.zero_grad()

        input = torch.tensor(inputs[batch * batch_size : batch * batch_size + batch_size])
        if params["model"] == "LSTM":
            input = np.swapaxes(input, 0, 1)

        label = torch.tensor(labels[batch * batch_size : batch * batch_size + batch_size])

        predictions = model(input).squeeze(0)
        
        loss = criterion(predictions.double(), label)
        acc = binary_accuracy(predictions.double(), label)

        if grad:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return (epoch_loss / int(len(inputs) / batch_size),
            epoch_acc / int(len(inputs) / batch_size))

def start_model():

    params = {
        "model": "MLP",
        "input_dim": 6 * 2 * 2 * 16,
        "hidden_dim": 100,
        "n_epochs": 10,
        "batch_size": 8,
        "output_dim": 50,
        "n_layers": 1,
        "bidirectional": False,
        "dropout": 0.
    }

    train_x, train_y, test_x, test_y = read_data(mlp=params["model"] == "MLP")

    if params["model"] == "MLP":
        model = MLP(params)
    else:
        model = RNN(params)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(params["n_epochs"]):

        train_loss, train_acc = train(
            model, params, train_x, train_y, optimizer, criterion, grad=True)
        print ("TRAIN Epoch: {}, Loss: {}, Accuracy: {}".format(epoch, train_loss, train_acc))

        test_loss, test_acc = train(
            model, params, test_x, test_y, optimizer, criterion, grad=False)
        print ("TEST Epoch: {}, Loss: {}, Accuracy: {}".format(epoch, test_loss, test_acc))

    l1, l2 = list(model.parameters())

def main():
    start_model()

main()
