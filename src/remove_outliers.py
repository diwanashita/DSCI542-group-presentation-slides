import pandas as pd
import sys
import os

#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# ChatGPT was used to ensure this function adheered to the guidlines mentioned in Milestone and 
# Breast Cancer Predictor example from Tiffany. 
def remove_outliers(data, column, lower, upper):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mask = (data[column] >= lower) & (data[column] <= upper)
    return data[mask].reset_index(drop=True)
