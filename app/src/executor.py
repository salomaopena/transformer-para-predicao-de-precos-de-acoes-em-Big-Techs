# Executor file
# Authors: Gabriel França de Almeida e Eneia Gazite
# Start Date: 2025-05-26
# Last Update: 2025-05-26

# Import section
import pandas as pd
import numpy as np

from datetime import datetime, date, timedelta

from dataset import Dataset
from pre_processor import cut_DataFrame_by_Period, convert_DateString_to_Date
from validator import is_String, is_DataFrame

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
        if is_DataFrame(dataset.dataFrame):
            # Iterate through the DataFrame rows
            for index, row in dataset.dataFrame.iterrows():
                # Try to convert the 'Date' column to a date object
                try:
                    # Validate if the 'Date' column is a string
                    if is_String(dataset.dataFrame.loc[index, 'Date']):
                        # Convert the 'Date' string to a date object
                        dataset.dataFrame.loc[index, 'Date'] = convert_DateString_to_Date(dataset.dataFrame['Date'][index])
                # Catch any TypeError that may occur during conversion
                except TypeError as error:
                    print("Error: ", error)
            # Cut the DataFrame by the last 90 days
            selectedData = cut_DataFrame_by_Period(dataset.dataFrame, 90)
            print(selectedData)
    except TypeError as error:
        print("Error: ", error)