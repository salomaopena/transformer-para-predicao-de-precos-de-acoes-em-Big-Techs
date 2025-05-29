# Validator file
# Authors: Gabriel França de Almeida e Eneia Gazite
# Start Date: 2025-05-26
# Last Update: 2025-05-26

# Import section
import pandas as pd

# Code section
def is_DataFrame(data: any) -> bool:
    if isinstance(data, pd.DataFrame):
        return True
    else:
        raise TypeError("Data is not a pandas DataFrame")