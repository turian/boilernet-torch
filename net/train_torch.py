#! /usr/bin/python3

import argparse
import csv
import math
import os
import pickle

import numpy as np

import torch
from sklearn.utils import class_weight
#from torchtext.data import Dataset, Example, Field

from leaf_classifier_torch import LeafClassifier
import torch.utils.data as data


class CustomDataset(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

def get_dataset(dataset_file, batch_size, shuffle=True, repeat=True):
    dataset = torch.load(dataset_file)
    custom_dataset = CustomDataset(dataset)
    if repeat:
        dataloader = data.DataLoader(custom_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=4, pin_memory=True, worker_init_fn=repeat_seed)
    else:
        dataloader = data.DataLoader(custom_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
    return dataloader

def repeat_seed(worker_id):
    worker_seed = torch.initial_seed() % (2**32 - 1)
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)




def _read_example(example):
    return {
        "doc_feature_list": example["doc_feature_list"],
        "doc_label_list": example["doc_label_list"],
    }


def get_class_weights(train_set_file):
    y_train = []
    for _, y in get_dataset(train_set_file, 1, False):
        y_train.extend(y.numpy().flatten())
    return class_weight.compute_class_weight(
        class_weight="balanced", classes=[0, 1], y=y_train
    )


def main():
    torch.autograd.set_detect_anomaly(True)
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "DATA_DIR", help="Directory of files produced by the preprocessing script"
    )
    ap.add_argument(
        "-l", "--num_layers", type=int, default=2, help="The number of RNN layers"
    )
    ap.add_argument(
        "-u",
        "--hidden_units",
        type=int,
        default=256,
        help="The number of hidden LSTM units",
    )
    ap.add_argument(
        "-d", "--dropout", type=float, default=0.5, help="The dropout percentage"
    )
    ap.add_argument(
        "-s", "--dense_size", type=int, default=256, help="Size of the dense layer"
    )
    ap.add_argument("-e", "--epochs", type=int, default=20, help="The number of epochs")
    ap.add_argument("-b", "--batch_size", type=int, default=16, help="The batch size")
    ap.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Calculate metrics and save the model after this many epochs",
    )
    ap.add_argument(
        "--working_dir", default="train", help="Where to save checkpoints and logs"
    )
    args = ap.parse_args()

    info_file = os.path.join(args.DATA_DIR, "info.pkl")
    with open(info_file, "rb") as fp:
        info = pickle.load(fp)
        train_steps = math.ceil(info["num_train_examples"] / args.batch_size)

    train_set_file = os.path.join(args.DATA_DIR, "train.pt")
    train_dataset = get_dataset(train_set_file, args.batch_size)

    dev_set_file = os.path.join(args.DATA_DIR, "dev.pt")
    if os.path.isfile(dev_set_file):
        dev_dataset = get_dataset(dev_set_file, 1, repeat=False)
    else:
        dev_dataset = None

    test_set_file = os.path.join(args.DATA_DIR, "test.pt")
    if os.path.isfile(test_set_file):
        test_dataset = get_dataset(test_set_file, 1, repeat=False)
    else:
        test_dataset = None

    class_weights = get_class_weights(train_set_file)
    print("using class weights {}".format(class_weights))

    kwargs = {
        "input_size": info["num_words"] + info["num_tags"],
        "hidden_size": args.hidden_units,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "dense_size": args.dense_size,
    }
    clf = LeafClassifier(**kwargs)

    ckpt_dir = os.path.join(args.working_dir, "ckpt")
    log_file = os.path.join(args.working_dir, "train.csv")
    os.makedirs(ckpt_dir, exist_ok=True)

    params_file = os.path.join(args.working_dir, "params.csv")
    print("writing {}...".format(params_file))
    with open(params_file, "w") as fp:
        writer = csv.writer(fp)
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])

    clf.train(
        train_dataset=train_dataset,
        train_steps=train_steps,
        epochs=args.epochs,
        optimizer=torch.optim.Adam(clf.parameters(), lr=1e-4),
        loss_function=torch.nn.functional.binary_cross_entropy,
        log_file=log_file,
        ckpt=ckpt_dir,
            class_weight=class_weights,
            dev_dataset=dev_dataset,
            dev_steps=info.get("num_dev_examples"),
                      test_dataset=test_dataset,
                      test_steps=info.get("num_test_examples"),
        interval=args.interval,
    )


if __name__ == "__main__":
    main()
