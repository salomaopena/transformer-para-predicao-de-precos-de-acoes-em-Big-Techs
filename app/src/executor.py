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

from transformer_processor.transformer import Transformer
from stock_predictor import StockPredictor
from data_processor.stock_dataset import StockDataset
from pre_processor.pre_processor import cut_dataFrame_by_period, convert_dateString_to_date
from pre_processor.validator import is_string, is_dataFrame

# Constants
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
TRAINING_END_DATE_STRING = '12/31/2024'
TRAINING_END_DATE = convert_dateString_to_date(TRAINING_END_DATE_STRING)
EVALUATION_END_DATE_STRING = '05/29/2025'
EVALUATION_END_DATE = convert_dateString_to_date(EVALUATION_END_DATE_STRING)

# Load datasets
APPL_PANDAS_DATAFRAME = pd.read_csv(r'./../data/Historical_Data_APPL_1Y.csv')
GOOGL_PANDAS_DATAFRAME = pd.read_csv(r'./../data/Historical_Data_GOOGL_1Y.csv')
IBM_PANDAS_DATAFRAME = pd.read_csv(r'./../data/Historical_Data_IBM_1Y.csv')
MFST_PANDAS_DATAFRAME = pd.read_csv(r'./../data/Historical_Data_MSFT_1Y.csv')

# Instantiate datasets
allPurposeDatasets = [ 
    StockDataset(title="APPL", dataFrame=APPL_PANDAS_DATAFRAME, windowSize=10), 
    StockDataset(title="GOOGL", dataFrame=GOOGL_PANDAS_DATAFRAME, windowSize=10), 
    StockDataset(title="IBM", dataFrame=IBM_PANDAS_DATAFRAME, windowSize=10),
    StockDataset(title="MSFT", dataFrame=MFST_PANDAS_DATAFRAME, windowSize=10),
]

trainingDatasets = []

evaluationDatasets = []

# Code section

# Pre-processing: Repeat for each dataset
for dataset in allPurposeDatasets:
    try:
        # Validate if the dataFrame is a pandas DataFrame
        if is_dataFrame(dataset.dataFrame):

            # Print the title of the dataset being processed
            print(f'Processing dataset: {dataset.title}')
            # Print the shape of the DataFrame
            # print(f'DataFrame shape: {dataset.__shape__()}')

            # Iterate through the DataFrame rows
            for index, row in dataset.dataFrame.iterrows():
                # Try to convert the 'Date' column to a date object
                try:
                    # Validate if the 'Date' column is a string
                    if is_string(dataset.dataFrame.loc[index, 'Date']):
                        # Convert the 'Date' string to a date object
                        dataset.dataFrame.loc[index, 'Date'] = convert_dateString_to_date(dataset.dataFrame['Date'][index])

                # Catch any TypeError that may occur during conversion
                except TypeError as error:
                    print("Error: ", error)

            # Create the training dataset by cutting the DataFrame
            trainingDatasets.append(StockDataset(title=("Training dataset - " + dataset.title),
                                                dataFrame=cut_dataFrame_by_period(dataset.dataFrame, days=90, endDate=TRAINING_END_DATE),
                                                windowSize=10))
            
             # Get the shape of the appended training dataset
            print(f"{trainingDatasets[-1].__shape__()}")

            #Create the evaluation dataset by cutting the DataFrame
            evaluationDatasets.append(cut_dataFrame_by_period(dataset.dataFrame, days=90, endDate=EVALUATION_END_DATE))

    except TypeError as error:
        print("Error: ", error)

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
    'vocabularySize': 63,
    'model': 32,
    'dropout': 0.1,
    'numberEncoderLayers': 1,
    'numberDecoderLayers': 1,
    'numberHeads': 9
}

# Define model here
model = Transformer(**args)

# Instantiate datasets
trainDataLoader = DataLoader(trainingDatasets[0], batch_size=256, collate_fn=collate_function)
evaluetionDataLoader = DataLoader(evaluationDatasets[0], batch_size=256, collate_fn=collate_function)

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
for epoch in range(1, 10):
    print(f"Epoch {epoch} of 9")
    startTime = time.time()
    trainLoss, trainAccumulator, historyLoss, historyAccumulator = train(model, optimizer, trainDataLoader, lossFunction, epoch)
    history['trainLoss'] += historyLoss
    history['trainAccumulator'] += historyAccumulator
    endTime = time.time()
    evaluationLoss, evaluationAccumulator, historyLoss, historyAccumulator = evaluate(model, evaluetionDataLoader, lossFunction)
    history['evaluationLoss'] += historyLoss
    history['evaluationAccumulator'] += historyAccumulator
    print((f"Epoch: {epoch}, Train loss: {trainLoss:.3f}, Train acc: {trainAccumulator:.3f}, Val loss: {evaluationLoss:.3f}, Val acc: {evaluationAccumulator:.3f} "f"Epoch time = {(endTime - startTime):.3f}s"))

stockPedictor = StockPredictor(model)