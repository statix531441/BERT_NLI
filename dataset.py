import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import *
from transformers import RobertaTokenizer

class LSTMDataset(Dataset):
    def __init__(self, df, opt):
        self.df = df
        self.vocab = load_vocab(opt)
        vocab_size = len(self.vocab)
        self.word_embeddings = nn.Embedding(vocab_size, opt.embedding_dim)
        self.word_embeddings.load_state_dict(torch.load(f'{opt.data_folder}/word_embeddings.pth'))

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        sentence1, sentence2 = self.df.loc[idx, ['sentence1', 'sentence2']]
        sentence1, sentence2 = clean_text(sentence1), clean_text(sentence2)
        
        y = le(self.df.loc[idx, 'gold_label'])
        y = torch.tensor(y)

        input_ids1 = torch.ones(100, dtype=torch.long)
        input_ids2 = torch.ones(100, dtype=torch.long)

        for i, word in enumerate(sentence1.split()):
            input_ids1[i] = self.vocab[word] if word in self.vocab else 0

        for i, word in enumerate(sentence2.split()):
            input_ids2[i] = self.vocab[word] if word in self.vocab else 0

        embeds1, embeds2 = self.word_embeddings(input_ids1), self.word_embeddings(input_ids2)
        return [embeds1, embeds2], y



class BERTCombinedDataset(Dataset):
    def __init__(self, df, opt):
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
        sentence1 = df['sentence1'].tolist()
        sentence2 = df['sentence2'].tolist()
        info = tokenizer(sentence1, sentence2, padding='max_length', max_length=200, return_tensors='pt')

        self.input_ids = info['input_ids']
        self.token_type_ids = info['token_type_ids']
        self.attention_mask = info['attention_mask']
        
        self.y = torch.tensor(df['gold_label'].apply(lambda label: le(label)))

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):

        input_ids = self.input_ids[idx]
        token_type_ids = self.token_type_ids[idx]
        attention_mask = self.attention_mask[idx]
        y = self.y[idx]

        return [input_ids, token_type_ids, attention_mask], y
    
class RobertaCombinedDataset(Dataset):
    def __init__(self, df, opt):
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        sentence1 = df['sentence1'].tolist()
        sentence2 = df['sentence2'].tolist()
        info = tokenizer(sentence1, sentence2, padding='max_length', max_length=200, return_tensors='pt')

        self.input_ids = info['input_ids']
        self.attention_mask = info['attention_mask']
        
        self.y = torch.tensor(df['gold_label'].apply(lambda label: le(label)))

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):

        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        y = self.y[idx]

        return [input_ids, attention_mask], y
    
class BERTSeperateDataset(Dataset):
    def __init__(self, df, opt):
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
        sentence1 = df['sentence1'].tolist()
        sentence2 = df['sentence2'].tolist()
        info1 = tokenizer(sentence1, padding='max_length', max_length=100, return_tensors='pt')
        info2 = tokenizer(sentence2, padding='max_length', max_length=100, return_tensors='pt')

        self.input_ids1 = info1['input_ids']
        # These 2 probably not required
        self.token_type_ids1 = info1['token_type_ids']
        self.attention_mask1 = info1['attention_mask'] 

        self.input_ids2 = info2['input_ids']
        # These 2 probably not required
        self.token_type_ids2 = info2['token_type_ids']
        self.attention_mask2 = info2['attention_mask'] 

        self.y = torch.tensor(df['gold_label'].apply(lambda label: le(label)))

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):

        input_ids1 = self.input_ids1[idx]
        token_type_ids1 = self.token_type_ids1[idx]
        attention_mask1 = self.attention_mask1[idx]

        input_ids2 = self.input_ids2[idx]
        token_type_ids2 = self.token_type_ids2[idx]
        attention_mask2 = self.attention_mask2[idx]

        y = self.y[idx]

        return [input_ids1, token_type_ids1, attention_mask1, input_ids2, token_type_ids2, attention_mask2], y

    
Datasets = {
    'LSTMDataset': LSTMDataset,
    'BERTCombinedDataset': BERTCombinedDataset,
    'BERTSeperateDataset': BERTSeperateDataset,
    'RobertaCombinedDataset': RobertaCombinedDataset,
}

if __name__ == "__main__":
    import os
    import pandas as pd
    from options import Options

    opt = Options(dataset='RobertaCombinedDataset', model="", tag="Test")
    opt.data_folder = 'data/RobertaCombinedDatasetTEST'
    os.makedirs(opt.data_folder, exist_ok=True)

    train_df = pd.read_json('original/snli_1.0_train.jsonl', lines=True)
    train_df = train_df.sample(10).reset_index(drop=True)

    ### LSTM Special Treatment in train.py (Modify FastEmbeddings)
    if opt.model == "LSTM":
        # Assume that if vocab is generated then embeddings are also generated and saved
        try:
            vocab = load_vocab(opt)
        except:
            print("Creating vocab")
            # Create vocabulary only from train_df
            vocab = create_vocab(train_df, opt)
            vocab_size = len(vocab)
            word_embeddings = nn.Embedding(vocab_size, opt.embedding_dim)

            # Replace this line with actual fasttext embeddings
            pretrained_fasttext_embeddings  = torch.rand((vocab_size, opt.embedding_dim))

            word_embeddings.weight.data.copy_(pretrained_fasttext_embeddings)
            torch.save(word_embeddings.state_dict(), f'{opt.data_folder}/word_embeddings.pth')
            print("Save vocab and word embeddings")

    Dataset = Datasets[opt.dataset]
    train_set = Dataset(train_df, opt)

    X, y = train_set[0]

    for x in X:
        print(x.shape)
    print(y)