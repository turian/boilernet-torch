#! /usr/bin/python3


import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from torch.autograd import Variable
from tqdm import tqdm


class Metrics:
    """Calculate metrics for a dev-/testset and add them to the logs."""
    def __init__(self, clf, data, steps, interval, prefix=''):
        self.clf = clf
        self.data = data
        self.steps = steps
        self.interval = interval
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs):
        if (epoch + 1) % self.interval == 0:
            y_true, y_pred = self.clf.evaluate(self.data, self.steps)
            #print(y_true, y_pred)
            p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average='binary')
        else:
            p, r, f, s = np.nan, np.nan, np.nan, np.nan
        logs_new = {'{}_precision'.format(self.prefix): p,
                    '{}_recall'.format(self.prefix): r,
                    '{}_f1'.format(self.prefix): f,
                    '{}_support'.format(self.prefix): s}
        logs.update(logs_new)


class Saver:
    """Save the model."""
    def __init__(self, path, interval):
        self.path = path
        self.interval = interval

    def on_epoch_end(self, epoch, model):
        if (epoch + 1) % self.interval == 0:
            file_name = os.path.join(self.path, 'model.{:03d}.pt'.format(epoch))
            torch.save(model.state_dict(), file_name)

class LeafClassifier(nn.Module):
    """This classifier assigns labels to sequences based on words and HTML tags."""

    def __init__(self, input_size, num_layers, hidden_size, dropout, dense_size):
        """Construct the network."""
        super(LeafClassifier, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dense_size = dense_size

        self.fc1 = nn.Linear(self.input_size, self.dense_size)
        self.masking = torch.zeros(1)
        self.rnn = nn.LSTM(
            input_size=self.dense_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=True,
        )
        self.fc2 = nn.Linear(
            self.hidden_size * 2, 1
        )  # Multiplied by 2 for bidirectionality
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = x * (x != self.masking).float()  # Masking
        output, _ = self.rnn(x)
        x = self.fc2(output)
        x = self.sigmoid(x)
        return x

    def do_train(self, train_dataset, train_steps, epochs, optimizer, loss_function, log_file=None, ckpt=None, class_weight=None, dev_dataset=None, dev_steps=None,
              test_dataset=None, test_steps=None, interval=1):
        """Train a number of input sequences."""
        metrics_dev = None
        metrics_test = None

        if dev_dataset is not None:
            metrics_dev = Metrics(self, dev_dataset, dev_steps, interval, 'dev')

        if test_dataset is not None:
            metrics_test = Metrics(self, test_dataset, test_steps, interval, 'test')

        saver = Saver(ckpt, interval)

        for epoch in range(epochs):
            total_loss = 0

            for b_x, b_y in train_dataset:
                # somehow this cast is necessary
                b_x = b_x.type(torch.FloatTensor)
                b_y = b_y.type(torch.FloatTensor)

                optimizer.zero_grad()
                output = self(b_x)
                #print(output.shape, b_y.shape)
                #output = output.view(-1)  # Flatten
                #print(output.shape, b_y.shape)
                loss = loss_function(output, b_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print("Epoch:", epoch, "Loss:", total_loss / train_steps)

            logs = {}

            if metrics_dev is not None:
                metrics_dev.on_epoch_end(epoch, logs)
                print("Dev metrics: ", logs)

            if metrics_test is not None:
                metrics_test.on_epoch_end(epoch, logs)
                print("Test metrics: ", logs)

            saver.on_epoch_end(epoch, self)

    def evaluate(self, dataset, steps):
        """Evaluate the model on the test data and return the metrics."""
        y_true, y_pred = [], []
        with torch.no_grad():
            for b_x, b_y in dataset:
                # Somehow this cast is necessary
                b_x = b_x.type(torch.FloatTensor)
                b_y = b_y.type(torch.FloatTensor)
                output = self(b_x)

                y_true.extend(b_y.cpu().numpy().flatten())
                y_pred.extend(torch.round(output).cpu().numpy().flatten())
        return np.array(y_true), np.array(y_pred)
