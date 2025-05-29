# Pre processor file
# Authors: Gabriel França de Almeida e Eneia Gazite
# Start Date: 2025-05-26
# Last Update: 2025-05-26

# Import section
import pandas as pd
import numpy as np

# Code section
def cut_DataFrame_by_Period(df: pd.DataFrame, days: int) -> pd.DataFrame:
    