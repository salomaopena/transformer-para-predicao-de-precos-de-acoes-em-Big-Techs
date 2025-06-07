# Executor file
# Authors: Gabriel França de Almeida e Eneia Gazite
# Start Date: 2025-05-26
# Last Update: 2025-05-26

# Import section
import pandas as pd
import numpy as np

from data_processor.dataset import Dataset
from pre_processor.pre_processor import cut_dataFrame_by_period, convert_dateString_to_date
from pre_processor.validator import is_string, is_dataFrame

# Load datasets
APPL_PANDAS_DATAFRAME = pd.read_csv(r'./../data/Historical_Data_APPL_1Y.csv')
GOOGL_PANDAS_DATAFRAME = pd.read_csv(r'./../data/Historical_Data_GOOGL_1Y.csv')
IBM_PANDAS_DATAFRAME = pd.read_csv(r'./../data/Historical_Data_IBM_1Y.csv')
MFST_PANDAS_DATAFRAME = pd.read_csv(r'./../data/Historical_Data_MSFT_1Y.csv')

datasets = [ 
    Dataset("APPL", APPL_PANDAS_DATAFRAME), 
    Dataset("GOOGL", GOOGL_PANDAS_DATAFRAME), 
    Dataset("IBM", IBM_PANDAS_DATAFRAME),
    Dataset("MSFT", MFST_PANDAS_DATAFRAME),
]

# Code section

# Pre-processing: Repeat for each dataset
for dataset in datasets:
    try:
        # Validate if the dataFrame is a pandas DataFrame
        if is_dataFrame(dataset.dataFrame):
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
            # Cut the DataFrame by the last 90 days
            selectedData = cut_dataFrame_by_period(dataset.dataFrame, 90)


            # Print the selected data
            # print(f"Selected data for {dataset.title}:")
            # print(selectedData)

    except TypeError as error:
        print("Error: ", error)

#Testing MultiHeadAttention
from transformer.multi_head_attention import MultiHeadAttention
from transformer.positional_enconding import PositionalEncoding

# multi_head_attention = MultiHeadAttention(headDimension=64, numberHeads=8)
positional_encoding = PositionalEncoding(model=64, dropoutProbability=0.1, maxLength=400)
