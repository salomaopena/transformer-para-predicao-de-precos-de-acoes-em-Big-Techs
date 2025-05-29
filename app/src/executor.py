# Executor file
# Authors: Gabriel França de Almeida e Eneia Gazite
# Start Date: 2025-05-26
# Last Update: 2025-05-26

# Import section
import pandas as pd
import numpy as np
import io, os, sys, types

from dataset import Dataset
from validator import is_DataFrame

# Load datasets
APPL_PANDAS_DATAFRAME = pd.read_csv(r'./../data/Historical_Data_APPL_6M.csv')
GOOGL_PANDAS_DATAFRAME = pd.read_csv(r'./../data/Historical_Data_GOOGL_6M.csv')
IBM_PANDAS_DATAFRAME = pd.read_csv(r'./../data/Historical_Data_IBM_6M.csv')
MFST_PANDAS_DATAFRAME = pd.read_csv(r'./../data/Historical_Data_MSFT_6M.csv')

datasets = [ 
    Dataset("APPL", APPL_PANDAS_DATAFRAME), 
    Dataset("GOOGL", GOOGL_PANDAS_DATAFRAME), 
    Dataset("IBM", IBM_PANDAS_DATAFRAME),
    Dataset("MSFT", MFST_PANDAS_DATAFRAME),
]

# Code section
for dataset in datasets:
    try:
        if is_DataFrame(dataset.dataFrame):
            print(dataset.dataFrame.head())  # Display the first few rows of the DataFrame
            print("\n")
    except TypeError as error:
        print("Error: ", error)