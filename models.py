import torch
import torch.nn as nn
from torch.functional import F

bert_tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')

class BERTCombinedModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.bert = bert_tokenizer
        self.linear = nn.Linear(768, 3)

    def forward(self, X):
        input_ids, token_type_ids, attention_mask = X

        # CLS representation of both sentences combined
        out = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        out = out[1]

        out = self.linear(out)
        return out
    
class BERTSeperateModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.bert = bert_tokenizer
        self.linear = nn.Linear(768 * 2, 3)

    def forward(self, X):
        input_ids1, token_type_ids1, attention_mask1, input_ids2, token_type_ids2, attention_mask2 = X

        # CLS representation of each sentence
        out1 = self.bert(input_ids1, token_type_ids=token_type_ids1, attention_mask=attention_mask1)
        out2 = self.bert(input_ids2, token_type_ids=token_type_ids2, attention_mask=attention_mask2)
        out1, out2 = out1[1], out2[1]
        
        out = torch.concat((out1, out2), dim=1)
        out = self.linear(out)
        return out
    
class LSTM(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.lstm = nn.LSTM(opt.embedding_dim, opt.hidden_dim, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(2*opt.hidden_dim, opt.classes)

    def forward(self, X):
        embeds1, embeds2 = X
        out1, _ = self.lstm(embeds1)
        out2, _ = self.lstm(embeds2)
        out1, out2 = out1[:,-1,:], out2[:,-1,:] # Take the final representations
        out = torch.concat((out1, out2), dim=1)
        out = self.linear(out)
        return out


Models = {
    'LSTM': LSTM,
    'BERTCombinedModel': BERTCombinedModel,
    'BERTSeperateModel': BERTSeperateModel,
}

if __name__ == "__main__":
    from options import Options
    from dataset import *
    from utils import *

    import pandas as pd
    from options import Options

    opt = Options(dataset='LSTMDataset', model="LSTM", tag="Test")
    opt.model = 'LSTM'
    opt.dataset = 'LSTMDataset'
    opt.data_folder = 'data/LSTM_TEST'

    df = pd.read_json('original/snli_1.0_train.jsonl', lines=True)
    df = df.sample(10).reset_index(drop=True)

    Dataset = Datasets[opt.dataset]
    test_set = Dataset(df, opt)
    train_loader = DataLoader(test_set, shuffle=True, batch_size=5)
    for batch_idx, (X, y) in enumerate(train_loader):
        break

    model = Models[opt.model](opt)
    out = model(X)
    print(out.shape)

    y_pred, y_test, test_accuracy = predict(model, train_loader)
    print(len(y_pred), len(y_test), test_accuracy)




