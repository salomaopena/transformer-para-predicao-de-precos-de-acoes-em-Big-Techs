# Validator file
# Authors: Gabriel França de Almeida e Eneia Gazite
# Start Date: 2025-05-26
# Last Update: 2025-05-26

# Import section
import pandas as pd

# Code section
def is_string(data: any) -> bool:
    if isinstance(data, str):
        return True
    else:
        raise TypeError("Data is not a string")

def is_even_number(number: int) -> bool:
    if (isinstance(number, int) and number % 2 == 0):
        return True
    else:
        raise ValueError("Number is not even")
    
def is_dataFrame(data: any) -> bool:
    if isinstance(data, pd.DataFrame):
        return True
    else:
        raise TypeError("Data is not a pandas DataFrame")