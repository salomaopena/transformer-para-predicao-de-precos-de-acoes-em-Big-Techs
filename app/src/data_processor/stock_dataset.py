import torch
from torch.utils.data import Dataset

class StockDataset(Dataset):
    def __init__(self, title, windowSize=10, dataFrame=None):
        super(StockDataset, self).__init__()
        self.title = title
        self.windowSize = windowSize
        self.dataFrame = dataFrame
        self.prices = (
            self.dataFrame['Close/Last']
            .astype(str)                     
            .str.strip()                     
            .replace('[\$,]', '', regex=True)
            .astype(float)
            .dropna()
            .values
        )
    def __shape__(self):
        return self.dataFrame.shape # Quantity of samples in the dataset

    def __len__(self):
        print(f"Dataset {self.title} has {len(self.prices)-self.windowSize} prices with window size {self.windowSize}.")
        return len(self.prices) - self.windowSize

    def __getitem__(self, idx):
        # Entrada: sequência de preços
        x = self.prices[idx:idx + self.windowSize]
        # Saída: mesma sequência deslocada (o que queremos prever)
        y = self.prices[idx + 1:idx + self.windowSize + 1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)