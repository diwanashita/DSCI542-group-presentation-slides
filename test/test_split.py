# tests/test_split.py
# References: Tiffany's Breast Cancer Predictor and ChatGPT were used as references

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pytest
import pandas as pd
from src.split import split


# Tests for split() function: 
# 1. Can it split a regular dataframe with string and int values
# 2. Test for empty dataset
# 3. Raise error when dataframe not in an existing directory
# 4. Raise error for invalid data type

# note: split and preprocessing will have similar tests. 


def test_split_normal_case(tmpdir):
    """
    Tests the function with a valid dataframe to checks if the output files 
    are created in a temporary directory.
    """
    # Create a sample df
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': ['A', 'B', 'C', 'D', 'E'],
        'target': [0, 1, 0, 1, 0]
    })
    
    # Use a temporary directory for saving files
    directory = tmpdir.mkdir("test_dir")
    
    # Call the split function
    split(data, 'target', str(directory))
    
    # Check if the files are created
    assert os.path.exists(os.path.join(directory, 'X_train.csv'))
    assert os.path.exists(os.path.join(directory, 'y_train.csv'))
    assert os.path.exists(os.path.join(directory, 'X_test.csv'))
    assert os.path.exists(os.path.join(directory, 'y_test.csv'))

def test_split_empty_dataframe():
    """
    Tests the function with an empty dataframe to ensure it raises a ValueError.
    """
    # Create an empty df
    data = pd.DataFrame()
    
    # Check if ValueError is raised
    with pytest.raises(ValueError, match="DataFrame must contain observations."):
        split(data, 'target', 'some_directory')

def test_split_nonexistent_directory():
    """
    Tests the function with a valid DataFrame but a non-existent directory to 
    ensure it raises a FileNotFoundError.
    """
    # Create a sample df
    data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'target': [0, 1, 0]
    })
    
    # Check if FileNotFoundError is raised
    with pytest.raises(FileNotFoundError, match="Directory nonexistent_directory does not exist."):
        split(data, 'target', 'nonexistent_directory')

def test_split_invalid_data_type():
    """
    Tests the function with a string instead of a DataFrame to ensure it raises a TypeError.
    """
    # Check if TypeError is raised for non-DataFrame input
    with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
        split("not_a_dataframe", 'target', 'some_directory')