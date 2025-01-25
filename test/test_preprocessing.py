# tests/test_preprocessing.py
# References: Tiffany's Breast Cancer Predictor and ChatGPT were used as references

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pytest
import pandas as pd
import tempfile
from src.preprocessing import preprocess

# Tests for preprocess:
# 1. With regular/normal df
# 2. With empty df
# 3. With invalid path 
# 4. With invalid input types

# note: split and preprocessing will have similar tests. 


def test_preprocess_normal():
    """
    Tests the function with valid DataFrames and checks if the output files are created
    """
    # Create sample df
    X_train = pd.DataFrame({
        'num_feature': [1, 2, 3],
        'cat_feature': ['A', 'B', 'A']
    })
    X_test = pd.DataFrame({
        'num_feature': [4, 5],
        'cat_feature': ['B', 'A']
    })
    numeric_features = ['num_feature']
    categorical_features = ['cat_feature']
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        preprocess(X_train, X_test, numeric_features, categorical_features, temp_dir)
        
        # Check if the processed files are created
        assert os.path.exists(os.path.join(temp_dir, 'delay_preprocessor.pickle'))
        assert os.path.exists(os.path.join(temp_dir, 'train_processed.pickle'))
        assert os.path.exists(os.path.join(temp_dir, 'test_processed.pickle'))

def test_preprocess_empty_dataframe():
    """
    Tests the function with empty DataFrames to ensure it raises a ValueError
    """
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    numeric_features = ['num_feature']
    categorical_features = ['cat_feature']
    
    with pytest.raises(ValueError, match="DataFrame must contain observations."):
        preprocess(X_train, X_test, numeric_features, categorical_features, 'some_path')

def test_preprocess_invalid_path():
    """
    Tests the function with a non-existent directory to ensure it raises a FileNotFoundError
    """
    X_train = pd.DataFrame({
        'num_feature': [1, 2, 3],
        'cat_feature': ['A', 'B', 'A']
    })
    X_test = pd.DataFrame({
        'num_feature': [4, 5],
        'cat_feature': ['B', 'A']
    })
    numeric_features = ['num_feature']
    categorical_features = ['cat_feature']
    
    with pytest.raises(FileNotFoundError, match="Directory some_invalid_path does not exist."):
        preprocess(X_train, X_test, numeric_features, categorical_features, 'some_invalid_path')

def test_preprocess_invalid_input_type():
    """
    Tests the function with an invalid input type (string instead of df) 
    to ensure it raises a TypeError
    """
    X_train = "not_a_dataframe"
    X_test = pd.DataFrame({
        'num_feature': [4, 5],
        'cat_feature': ['B', 'A']
    })
    numeric_features = ['num_feature']
    categorical_features = ['cat_feature']
    
    with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
        preprocess(X_train, X_test, numeric_features, categorical_features, 'some_path')