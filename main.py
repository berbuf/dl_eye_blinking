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

    embeddings, name = [], []
    for f in os.listdir(data_dir):
        if f.find(".mat") != -1:
            v = scipy.io.loadmat(data_dir + f)["data"].astype(np.float32)
            if len(v) > 30:
                v = v[:30]
            embeddings.append(v)
            name.append(f)
            print (f, v.shape)
    max_label = len(embeddings)
    labels = []
    for i in range(len(embeddings)):
        labels += [i] * len(embeddings[i])
    embeddings = np.concatenate(embeddings)
    if mlp:
        input = embeddings.reshape((len(embeddings), params["series_dim"] * params["input_dim"]))
    else:
        input = embeddings.reshape((len(embeddings), params["series_dim"], params["input_dim"]))
    return split_data(input, labels), max_label, name

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
        input = np.swapaxes(input, 0, 1)
        output, (h, c) = self.rnn(self.dropout(input))
        return self.log_softmax(self.fc(self.dropout(h[-1])).squeeze(0))

class MLP(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=params["series_dim"] * params["input_dim"],
                      out_features=params["hidden_dim"]),
            nn.Dropout(params["dropout"]),
            nn.ReLU(),
            nn.Linear(in_features=params["hidden_dim"], out_features=params["hidden_dim"]),
            nn.Dropout(params["dropout"]),
            nn.ReLU(),
            nn.Linear(in_features=params["hidden_dim"], out_features=params["output_dim"]),
        )
        self.softmax = nn.Softmax()

    def forward(self, input):
        output = self.model(input).squeeze(0)
        return self.softmax(output)

def binary_accuracy(preds, y, max_label=-1, mat_conf=False):
    _, preds = preds.max(1)
    correct = (preds == y).float()
    conf = None
    if mat_conf:
        conf = np.zeros((max_label, max_label))
        for i in range(len(preds)):
            conf[y[i], preds[i]] += 1
    return correct.sum() / len(correct), conf

def train(model, params, inputs, labels, optimizer, criterion, grad=False):

    model.train() if grad else model.eval()

    epoch_loss, epoch_acc = 0, 0
    batch_size = params["batch_size"]

    for batch in range(int(len(inputs) / batch_size)):
        if grad:
            optimizer.zero_grad()

        input = torch.tensor(inputs[batch * batch_size : batch * batch_size + batch_size])
        label = torch.tensor(labels[batch * batch_size : batch * batch_size + batch_size])

        predictions = model(input)
        predictions = torch.log(predictions).squeeze(0)

        loss = criterion(predictions.double(), label)
        acc, _ = binary_accuracy(predictions.double(), label)

        if not grad:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return (epoch_loss / int(len(inputs) / batch_size),
            epoch_acc / int(len(inputs) / batch_size))

def roc_accuracy(probs, y, roc, threshold):
    probs = probs.detach().numpy()

    val = (probs >= threshold).astype(np.int8)
    for v, id in zip(val, y):
        roc["true_pos"][id] += v[id]
        roc["false_neg"][id] += int(v[id] == 0)

    false_pos = (probs >= threshold).astype(np.int8)
    false_pos[range(len(probs)), y.numpy()] = 0
    false_pos = np.sum(false_pos, axis=0)

    true_neg = (probs < threshold).astype(np.int8)
    true_neg[range(len(probs)), y.numpy()] = 0
    true_neg = np.sum(true_neg, axis=0)

    roc["false_pos"] += false_pos
    roc["true_neg"] += true_neg

    return roc

def compute_roc(model, x, y, max_label):
    prob = model(torch.tensor(x))
    _, count = np.unique(y, return_counts=True)
    t_pos, f_neg, f_pos, t_neg = [], [], [], []
    for threshold in range(0, 11):
        roc = { "true_pos": np.zeros(max_label),
                "false_neg": np.zeros(max_label),
                "false_pos": np.zeros(max_label),
                "true_neg": np.zeros(max_label) }
        roc = roc_accuracy(prob, torch.tensor(y), roc, threshold / 10)
        t_pos.append(roc["true_pos"] / count)
        f_neg.append(roc["false_neg"] / count)
        f_pos.append(roc["false_pos"] / (len(x) - count))
        t_neg.append(roc["true_neg"] / (len(x) - count))

    print ("true positive")
    print (np.array(t_pos))
    print ("false positive")
    print (np.array(f_pos))
    print ("false negative")
    print (np.array(f_neg))
    print ("true negative")
    print (np.array(t_neg))

def start_model():

    params = {
        "model": "MLP",
        "series_dim": 16,
        "input_dim": 6 * 2 * 2,
        "hidden_dim": 200,
        "n_epochs": 500,
        "batch_size": 50,
        "dropout": 0.
    }

    (train_x, train_y, test_x, test_y), max_label, name = read_data(params,
                                                                    mlp=params["model"] == "MLP")
    params["output_dim"] = max_label

    model = MLP(params) if params["model"] == "MLP" else RNN(params)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(params["n_epochs"]):

        train_loss, train_acc = train(
            model, params, train_x, train_y, optimizer, criterion, grad=True)
        print ("TRAIN Epoch: {}, Loss: {}, Accuracy: {}".format(epoch, train_loss, train_acc))

        test_loss, test_acc = train(
            model, params, test_x, test_y, optimizer, criterion, grad=False)
        print ("TEST Epoch: {}, Loss: {}, Accuracy: {}".format(epoch, test_loss, test_acc))

    compute_roc(model, test_x, test_y, max_label)

    prob = model(torch.tensor(test_x))
    predictions = torch.log(prob).squeeze(0)
    acc, conf = binary_accuracy(predictions.double(), torch.tensor(test_y),
                                     max_label=max_label, mat_conf=True)
    print ("final acc", acc)
    """
    print (name)
    print (conf)
    print (np.unique(test_y, return_counts=True))
    """


def main():
    start_model()

main()
