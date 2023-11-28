import re
import string
import pandas as pd

import torch
from torch.functional import F

from collections.abc import Iterable

# Loads from original and saves changes into data/
def create_split(opt):
    train_df = pd.read_json('original/snli_1.0_train.jsonl', lines=True)
    test_df = pd.read_json('original/snli_1.0_test.jsonl', lines=True)
    dev_df = pd.read_json('original/snli_1.0_dev.jsonl', lines=True)

    train_df = train_df.sample(opt.train_size, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(opt.test_size, random_state=42).reset_index(drop=True)

    train_df.to_csv(f'{opt.data_folder}/train.csv', index=False)
    test_df.to_csv(f'{opt.data_folder}/test.csv', index=False)

    return train_df, test_df

def create_vocab(train_df, opt):
    vocab = {
        '<unk>': 0,
        '<pad>': 1,
    }
    sentences = pd.concat((train_df['sentence1'], train_df['sentence2']), axis=0).reset_index(drop=True)

    for i, sentence in enumerate(sentences):
        sent_split = clean_text(sentence).split()
        for word in sent_split:
            if word not in vocab:
                vocab[word] = len(vocab)

    with open(f'{opt.data_folder}/vocab.json', 'w') as f:
        import json
        json.dump(vocab, f, indent=2)

    return vocab

def load_vocab(opt):
    with open(f'{opt.data_folder}/vocab.json', 'r') as f:
        import json
        vocab = json.load(f)
        return vocab

def le(gold):
    if gold=='neutral': return 1
    elif gold=='entailment': return 2
    else: return 0

def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


def fit(model, train_loader, lossFn, optimizer):  
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    train_loss = 0
    train_accuracy = 0

    model.train()
    for batch_idx, (X, y) in enumerate(train_loader):
        X = [x.to(device) for x in X] if isinstance(X, Iterable) else X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = lossFn(pred, y)
        train_loss += loss.item()
        train_accuracy += torch.sum(F.softmax(pred, dim=1).argmax(axis=1) == y).item()
        loss.backward()
        optimizer.step()
    train_accuracy /= len(train_loader.dataset)
    return train_loss, train_accuracy

def predict(model, test_loader):
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    y_test = []
    y_pred = []
    test_accuracy = 0
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(test_loader):
            X = [x.to(device) for x in X] if isinstance(X, Iterable) else X.to(device)
            pred = model(X)
            y_test.extend(y.tolist())
            y_pred.extend(F.softmax(pred, dim=1).argmax(axis=1).tolist())
            test_accuracy += torch.sum(F.softmax(pred, dim=1).argmax(axis=1) == y).item()
    test_accuracy /= len(test_loader.dataset)
    return y_pred, y_test, test_accuracy