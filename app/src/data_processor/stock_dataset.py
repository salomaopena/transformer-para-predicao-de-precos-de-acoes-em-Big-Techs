from torch.utils.data import Dataset
class StockDataset(Dataset):
    def __init__(self, title, paddingIdx=0, startOfSentenceIdx=0, endOfSentenceIdx=0, dataFrame=None):
        super(StockDataset, self).__init__()
        self.title = title
        self.paddingIdx = paddingIdx
        self.startOfSentenceIdx = startOfSentenceIdx
        self.endOfSentenceIdx = endOfSentenceIdx
        self.dataFrame = dataFrame

    def __shape__(self):
        return self.dataFrame.shape # Quantity of samples in the dataset
    
    def __getitem__(self, index):
        return self.dataFrame['Close/Last'][index].rstrip("\n")