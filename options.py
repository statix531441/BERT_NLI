import json
import os
class Options:
    def __init__(self, dataset='BERTCombinedDataset', model="BERTCombinedModel", tag="Test"):
    
        # Initialize
        self.dataset = dataset
        self.model = model
        self.tag = tag
        self.train_size = 10000
        self.test_size = 10000

        # Train and test options
        self.epochs = 50
        self.batch_size = 50
        self.lr = 1e-4
        self.classes = 3
        self.finetune_bert_last_layer = False

        # Folders
        self.data_folder = f"data/{self.train_size}_{self.test_size}"
        self.model_folder = f"models/{model}" + f"_{tag}" + f"{'_finetune' if self.finetune_bert_last_layer else ''}"

        # Activate Dataset Options
        if dataset == 'BERTCombinedDataset':
            self.BERTCombinedDataset()
        elif dataset == 'LSTMDataset':
            self.LSTMDataset()
        
        # Activate Model Options
        if model == 'BERTCombinedModel':
            self.BERTCombinedModel()

    # Dataset Options
    def BERTCombinedDataset(self):
        pass

    def BERTSeperateDataset(self):
        pass

    def LSTMDataset(self):
        self.embedding_dim = 80
        self.hidden_dim = 50

    # Model Options
    def BERTCombinedModel(self):
        pass

    def save_options(self, folder):
        os.makedirs(f'{folder}', exist_ok=True)
        with open(f'{folder}/options.json', 'w') as f:
            json.dump(self.__dict__, f, indent=2)

    def load_options(self, folder):
        with open(f'{folder}/options.json', 'r') as f:
            self.__dict__.update(json.load(f))

if __name__ == "__main__":
    opt = Options(dataset='ECG', model='SmoothCNN', tag="Something")
    for k, v in opt.__dict__.items():
        print(k, v)
