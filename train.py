# Defaults
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('dark_background')

# PyTorch
import torch
import torch.nn as nn
from torch.functional import F

# Local
from dataset import *
from models import *
from options import Options
from utils import *


# Utilities
import os
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device, " used for training")

##################################################################

### Settings changes here (Use argparse)
opt = Options(dataset='BERTSeperateDataset', model="BERTSeperateModel", tag="Test")
opt.finetune_bert_last_layer = False




### Name and create model folder (Replace opt. with args. once argparse is implemented)
if opt.model == 'LSTM':
    opt.model_folder = f"models/{opt.model}" + f"_{opt.tag}"
opt.model_folder = f"models/{opt.model}" + f"_{opt.tag}" + f"{'_finetune' if opt.finetune_bert_last_layer else ''}"
os.makedirs(opt.model_folder, exist_ok=True)

### Update opt with changes, Initialize directories and save options
opt.data_folder = f"data/{opt.train_size}_{opt.test_size}"
os.makedirs(opt.data_folder, exist_ok=True)
for item in opt.__dict__.items():
    print(item)
opt.save_options(opt.model_folder)

### 🔐Load train.csv and test.csv
try:
    train_df = pd.read_csv(f'{opt.data_folder}/train.csv')
    test_df = pd.read_csv(f'{opt.data_folder}/test.csv')
except:
    print(f"Creating train, test split in {opt.data_folder}")
    train_df, test_df = create_split(opt)
    # 🤔

### LSTM Special Treatment (Modify FastEmbeddings)
if opt.model == "LSTM":
    # Assume that if vocab is generated then embeddings are also generated and saved
    try:
        vocab = load_vocab(opt)
    except:
        # Create vocabulary only from train_df
        vocab = create_vocab(train_df, opt)
        vocab_size = len(vocab)
        word_embeddings = nn.Embedding(vocab_size, opt.embedding_dim)
        pretrained_fasttext_embeddings  = torch.rand((vocab_size, opt.embedding_dim))

        word_embeddings.weight.data.copy_(pretrained_fasttext_embeddings)
        torch.save(word_embeddings.state_dict(), f'{opt.data_folder}/word_embeddings.pth')


### 🔐Select dataset and create dataloaders
Dataset = Datasets[opt.dataset]
train_set = Dataset(train_df, opt)
test_set = Dataset(test_df, opt)
train_loader = DataLoader(train_set, shuffle=True, batch_size=opt.batch_size)
test_loader = DataLoader(test_set, shuffle=True, batch_size=opt.batch_size)

### 🔐Select and load model
model = Models[opt.model](opt).to(device)

### 🧡Tuning Model Parameters
if opt.model in ('BERTCombinedModel', 'BERTSeperateModel'):
    for param in model.bert.parameters():
        param.requires_grad = False

    if opt.finetune_bert_last_layer:
        for param in model.bert.pooler.parameters():
            param.requires_grad = True

### Initialize loss, optimizer and train history
lossFn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
history = {
    "train_loss": [],
    "train_accuracy": [],
    "test_accuracy": []
}

### 🔐Train Loop 
for epoch in tqdm(range(opt.epochs)):

    train_loss, train_accuracy = fit(model, train_loader, lossFn, optimizer)
    _, _, test_accuracy = predict(model, test_loader)

    ############### Save Model ################
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    if len(history['test_accuracy']) and test_accuracy >= max(history['test_accuracy']):
        torch.save(state, f"{opt.model_folder}/best.pth")
    history['train_loss'].append(train_loss)
    history['train_accuracy'].append(train_accuracy)
    history['test_accuracy'].append(test_accuracy)

    torch.save(history, f"{opt.model_folder}/history.pth")
    torch.save(state, f"{opt.model_folder}/latest.pth")
    ###########################################