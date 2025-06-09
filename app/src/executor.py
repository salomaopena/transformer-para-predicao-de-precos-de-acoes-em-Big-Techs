# Executor file
# Authors: Gabriel França de Almeida e Eneia Gazite
# Start Date: 2025-05-26
# Last Update: 2025-05-26

# Import section
import pandas as pd
import numpy as np
import torch
import time
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from mpl_toolkits.axes_grid1 import ImageGrid

from data_processor.stock_dataset import StockDataset
from pre_processor.pre_processor import cut_dataFrame_by_period, convert_dateString_to_date
from pre_processor.validator import is_string, is_dataFrame
from transformer.transformer import Transformer

# Constants
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
TRAINING_END_DATE_STRING = '2024-12-31'
TRAINING_END_DATE = convert_dateString_to_date(TRAINING_END_DATE_STRING)
EVALUATION_END_DATE_STRING = '2025-05-29'
EVALUATION_END_DATE = convert_dateString_to_date(EVALUATION_END_DATE_STRING)

# Load datasets
APPL_PANDAS_DATAFRAME = pd.read_csv(r'./../data/Historical_Data_APPL_1Y.csv')
GOOGL_PANDAS_DATAFRAME = pd.read_csv(r'./../data/Historical_Data_GOOGL_1Y.csv')
IBM_PANDAS_DATAFRAME = pd.read_csv(r'./../data/Historical_Data_IBM_1Y.csv')
MFST_PANDAS_DATAFRAME = pd.read_csv(r'./../data/Historical_Data_MSFT_1Y.csv')

# Instantiate datasets
allPurposeDatasets = [ 
    StockDataset(title="APPL", dataFrame=APPL_PANDAS_DATAFRAME, paddingIdx=PAD_IDX, startOfSentenceIdx=SOS_IDX, endOfSentenceIdx=EOS_IDX), 
    StockDataset(title="GOOGL", dataFrame=GOOGL_PANDAS_DATAFRAME, paddingIdx=PAD_IDX, startOfSentenceIdx=SOS_IDX, endOfSentenceIdx=EOS_IDX), 
    StockDataset(title="IBM", dataFrame=IBM_PANDAS_DATAFRAME, paddingIdx=PAD_IDX, startOfSentenceIdx=SOS_IDX, endOfSentenceIdx=EOS_IDX),
    StockDataset(title="MSFT", dataFrame=MFST_PANDAS_DATAFRAME, paddingIdx=PAD_IDX, startOfSentenceIdx=SOS_IDX, endOfSentenceIdx=EOS_IDX),
]

trainingDatasets = [
    cut_dataFrame_by_period(allPurposeDatasets[0], days=90, endDate=TRAINING_END_DATE), 
    cut_dataFrame_by_period(allPurposeDatasets[1], days=90, endDate=TRAINING_END_DATE), 
    cut_dataFrame_by_period(allPurposeDatasets[2], days=90, endDate=TRAINING_END_DATE),
    cut_dataFrame_by_period(allPurposeDatasets[3], days=90, endDate=TRAINING_END_DATE),
]

evaluationDatasets = [
    cut_dataFrame_by_period(allPurposeDatasets[0], days=90, endDate=EVALUATION_END_DATE),
    cut_dataFrame_by_period(allPurposeDatasets[1], days=90, endDate=EVALUATION_END_DATE), 
    cut_dataFrame_by_period(allPurposeDatasets[2], days=90, endDate=EVALUATION_END_DATE),
    cut_dataFrame_by_period(allPurposeDatasets[3], days=90, endDate=EVALUATION_END_DATE),
]

# Code section

# Pre-processing: Repeat for each dataset
for dataset in allPurposeDatasets:
    try:
        # Validate if the dataFrame is a pandas DataFrame
        if is_dataFrame(dataset.dataFrame):

            # Print the title of the dataset being processed
            print(f'Processing dataset: {dataset.title}')
            # Print the shape of the DataFrame
            print(f'DataFrame shape: {dataset.__shape__()}')

            # Iterate through the DataFrame rows
            for index, row in dataset.dataFrame.iterrows():
                # Try to convert the 'Date' column to a date object
                try:
                    # Validate if the 'Date' column is a string
                    if is_string(dataset.dataFrame.loc[index, 'Date']):
                        # Convert the 'Date' string to a date object
                        dataset.dataFrame.loc[index, 'Date'] = convert_dateString_to_date(dataset.dataFrame['Date'][index])

                        # Print the value of 'Close/Last' for the current index
                        print(f'Item for index {index}: {dataset.__getitem__(index)}')

                # Catch any TypeError that may occur during conversion
                except TypeError as error:
                    print("Error: ", error)
            # Cut the DataFrame by the last 90 days
            selectedData = cut_dataFrame_by_period(dataset.dataFrame, days=90)

            # Print the selected data
            # print(f"Selected data for {dataset.title}:")
            # print(selectedData)

    except TypeError as error:
        print("Error: ", error)

# Training section

# Define a prediction method for the Transformer model
def predict(
        self,
        x: torch.Tensor,
        startOfSentenceIdx: int=1,
        endOfSentenceIdx: int=2,
        maxLength: int=None
    ) -> torch.Tensor:
    """
    Method to use at inference time. Predict y from x one token at a time. This method is greedy
    decoding.
    Input
        x: str
    Output
        (B, L, C) logits
    """

    # Pad the tokens with beginning and end of sentence tokens
    x = torch.cat([
        torch.tensor([startOfSentenceIdx]), 
        x, 
        torch.tensor([endOfSentenceIdx])]
    ).unsqueeze(0)

    encoder_output, mask = self.transformer.encode(x) # (B, S, E)
    
    if not maxLength:
        maxLength = x.size(1)

    outputs = torch.ones((x.size()[0], maxLength)).type_as(x).long() * startOfSentenceIdx
    for step in range(1, maxLength):
        y = outputs[:, :step]
        probs = self.transformer.decode(y, encoder_output)
        output = torch.argmax(probs, dim=-1)
        
        # Prediction step by step
        # print(f"Knowing {y} we output {output[:, -1]}")

        if output[:, -1].detach().numpy() in (endOfSentenceIdx, startOfSentenceIdx):
            break
        outputs[:, step] = output[:, -1]
        
    return outputs

# Define a train method for the Transformer model
def train(model, optimizer, loader, lossFunction, epoch):
    model.train()
    losses = 0
    accumulator = 0
    historyLoss = []
    historyAccumulator = [] 

    with tqdm(loader, position=0, leave=True) as tepoch:
        for x, y in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

        optimizer.zero_grad()
        logits = model(x, y[:, :-1])
        loss = lossFunction(logits.contiguous().view(-1, model.vocab_size), y[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        losses += loss.item()
        
        preds = logits.argmax(dim=-1)
        masked_pred = preds * (y[:, 1:]!=PAD_IDX)
        accuracy = (masked_pred == y[:, 1:]).float().mean()
        acc += accuracy.item()
        
        historyLoss.append(loss.item())
        historyAccumulator.append(accuracy.item())
        tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy.item())

    return losses / len(list(loader)), accumulator / len(list(loader)), historyLoss, historyAccumulator

# Define a evaluate method for the Transformer model
def evaluate(model, loader, lossFunction):
    model.eval()
    losses = 0
    accumulator = 0
    historyLoss = []
    historyAccumulator = [] 

    for x, y in tqdm(loader, position=0, leave=True):

        logits = model(x, y[:, :-1])
        loss = lossFunction(logits.contiguous().view(-1, model.vocab_size), y[:, 1:].contiguous().view(-1))
        losses += loss.item()
        
        preds = logits.argmax(dim=-1)
        masked_pred = preds * (y[:, 1:]!=PAD_IDX)
        accuracy = (masked_pred == y[:, 1:]).float().mean()
        acc += accuracy.item()
        
        historyLoss.append(loss.item())
        historyAccumulator.append(accuracy.item())

    return losses / len(list(loader)), acc / len(list(loader)), historyLoss, historyAccumulator

# Define a collate function for the Transformer model, with DataLoader
def collate_function(batch):
    """ 
    This function pads inputs with PAD_IDX to have batches of equal length
    """
    sourceBatch, targetBatch = [], []
    for sourceSample, targetSample in batch:
        sourceBatch.append(sourceSample)
        targetBatch.append(targetSample)

    sourceBatch = pad_sequence(sourceBatch, padding_value=PAD_IDX, batch_first=True)
    targetBatch = pad_sequence(targetBatch, padding_value=PAD_IDX, batch_first=True)
    return sourceBatch, targetBatch

# Model hyperparameters
args = {
    'vocabularySize': 128,
    'model': 128,
    'dropout': 0.1,
    'numberEncoderLayers': 1,
    'numberDecoderLayers': 1,
    'numberHeads': 4
}

# Define model here
model = Transformer(**args)

# Instantiate datasets
trainDataLoader = DataLoader(trainingDatasets[0], batch_size=256, collate_fn=collate_function)
evaluetionDataLoader = DataLoader(evaluationDatasets[0], batch_size=256, collate_fn=collate_function)

# During debugging, we ensure sources and targets are indeed reversed
# s, t = next(iter(dataloader_train))
# print(s[:4, ...])
# print(t[:4, ...])
# print(s.size())

# Initialize model parameters
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

# Define loss function : we ignore logits which are padding tokens
lossFunction = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

# Create history to dictionary
history = {
    'trainLoss': [],
    'evaluationLoss': [],
    'trainAccumulator': [],
    'evaluationAccumulator': []
}

# Main loop
for epoch in range(1, 4):
    startTime = time.time()
    trainLoss, trainAccumulator, historyLoss, historyAccumulator = train(model, optimizer, trainDataLoader, lossFunction, epoch)
    history['trainLoss'] += historyLoss
    history['trainAccumulator'] += historyAccumulator
    endTime = time.time()
    evaluationLoss, evaluationAccumulator, historyLoss, historyAccumulator = evaluate(model, evaluetionDataLoader, lossFunction)
    history['evaluationLoss'] += historyLoss
    history['evaluationAccumulator'] += historyAccumulator
    print((f"Epoch: {epoch}, Train loss: {trainLoss:.3f}, Train acc: {trainAccumulator:.3f}, Val loss: {evaluationLoss:.3f}, Val acc: {evaluationAccumulator:.3f} "f"Epoch time = {(endTime - startTime):.3f}s"))
