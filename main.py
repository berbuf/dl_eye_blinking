from __future__ import print_function

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import scipy.io

def split_data(sentences, labels):
    np.random.seed(12)
    split_frac = 0.8
    len_feat = len(sentences)
    indexes = np.random.permutation(len_feat)
    sentences = np.array(sentences)[indexes]
    labels = np.array(labels)[indexes]
    return (sentences[0:int(split_frac*len_feat)],
            labels[0:int(split_frac*len_feat)],
            sentences[int(split_frac*len_feat):],
            labels[int(split_frac*len_feat):])

def read_data(params, data_dir="./data/", mlp=False):
    embeddings = [ scipy.io.loadmat(data_dir + f)["data"].astype(np.float32)
                   for f in os.listdir(data_dir) if f.find(".mat") != -1 ]
    max_label = len(embeddings)
    labels = []
    for i in range(len(embeddings)):
        labels += [i] * len(embeddings[i])
    embeddings = np.concatenate(embeddings)
    if mlp:
        input = embeddings.reshape((len(embeddings), params["series_dim"] * params["input_dim"]))
    else:
        input = embeddings.reshape((len(embeddings), params["series_dim"], params["input_dim"]))
    return split_data(input, labels), max_label

class RNN(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.rnn = nn.LSTM(params["input_dim"],
                           params["hidden_dim"],
                           num_layers=1,
                           bidirectional=False,
                           dropout=params["dropout"])

        self.fc = nn.Linear(params["hidden_dim"], params["output_dim"])
        self.log_softmax = nn.LogSoftmax()
        self.dropout = nn.Dropout(params["dropout"])

    def forward(self, input):
        # input = [sent len, batch size]
        input = self.dropout(input)
        output, (h, c) = self.rnn(input)
        return self.log_softmax(self.fc(self.dropout(h[-1])).squeeze(0))

class MLP(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=params["series_dim"] * params["input_dim"],
                      out_features=params["hidden_dim"]),
            nn.Dropout(params["dropout"]),
            nn.ReLU(),
            nn.Linear(in_features=params["hidden_dim"], out_features=params["output_dim"]),
        )
        self.dropout = nn.Dropout(params["dropout"])
        self.log_softmax = nn.LogSoftmax()

    def forward(self, input):
        return self.log_softmax(self.model(input).squeeze(0))

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
        "model": "LSTM",
        "series_dim": 16,
        "input_dim": 6 * 2 * 2,
        "hidden_dim": 100,
        "n_epochs": 500,
        "batch_size": 50,
        "dropout": 0.
    }

    (train_x, train_y, test_x, test_y), max_label = read_data(params, mlp=params["model"] == "MLP")
    params["output_dim"] = max_label

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

    print (test_y)
def main():
    start_model()

main()
