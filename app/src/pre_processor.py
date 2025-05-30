# Pre processor file
# Authors: Gabriel França de Almeida e Eneia Gazite
# Start Date: 2025-05-26
# Last Update: 2025-05-26

# Import section
import pandas as pd
import numpy as np

from datetime import datetime, date, timedelta

# Code section
def convert_DateString_to_Date(dateString: str) -> date:
    """
    Converts a date string in the format 'MM/DD/YYYY' to a date object.
    """
    return datetime.strptime(dateString, '%m/%d/%Y').date()



def cut_DataFrame_by_Period(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """
    Cuts the DataFrame from today date to the days quantity.
    """
    endDate = date.today()
    startDate = endDate - timedelta(days=days)
    return df.loc[(df['Date'] >= startDate) & (df['Date'] <= endDate)] 