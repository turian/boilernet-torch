#!/usr/bin/env python3

import json
import argparse
import pickle
from collections import defaultdict

import nltk
from bs4 import BeautifulSoup

from preprocess_torch import process, get_doc_inputs, read_file
from leaf_classifier_torch import LeafClassifier
import torch

def load_classifier(checkpoint_path, params):
    checkpoint = torch.load(checkpoint_path)
    print(checkpoint.keys())
    #model = LeafClassifier(**checkpoint['model_args'])
    model = LeafClassifier(**params)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def preprocess_single_file(html_file, word_map, tag_map):
    with open(html_file, 'rb') as hfile:
        doc = BeautifulSoup(hfile, features='html5lib')
    tags = defaultdict(int)
    words = defaultdict(int)
    processed_doc = process(doc, tags, words)

    return get_doc_inputs([processed_doc], word_map, tag_map)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('HTML_FILE', help='Path to the HTML file to preprocess')
    ap.add_argument('--info', help='Info pickle file from preprocess_torch.py', required=True)
    ap.add_argument('--checkpoint', help='LeafClassifier checkpoint', required=True)
    ap.add_argument('--kwargs', help='LeafClassifier kwargs JSON.', required=True)
    args = ap.parse_args()

    # files required for tokenization
    nltk.download('punkt')

    with open(args.info, 'rb') as fp:
        info = pickle.load(fp)

    word_map = info["word_map"]
    tag_map = info["tag_map"]

    kwargs = json.load(open(args.kwargs))
    print(kwargs)
    model = load_classifier(args.checkpoint, kwargs)

    preprocessed_data = preprocess_single_file(args.HTML_FILE, word_map, tag_map)

    for features, labels in preprocessed_data:
        print("Features: ", features)
        print("Labels: ", labels)
        with torch.no_grad():
            output = model(features.float())
            print("Output: ", output)
            print("Predicted: ", torch.argmax(output, dim=1))

if __name__ == '__main__':
    main()