#!/usr/bin/env python
# coding: utf-8

# Data loading and basic cleaning

# In[6]:


import pandas as pd
import os
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


def load_and_clean_data(file_path):
    """
    Load data and perform basic cleaning.
    """
    ttc = pd.read_csv(file_path, parse_dates=['Date'])
    ttc['Time'] = pd.to_datetime(ttc['Time']).dt.time
    ttc['Date_'] = ttc['Date'].dt.date
    ttc['Month'] = ttc['Date'].dt.month
    ttc['Hour'] = ttc['Time'].map(lambda x: x.hour)
    ttc = ttc.drop(columns=['Date', 'Time', 'Direction', 'Vehicle']).dropna()
    return ttc


# Other functions for your reference

# """
# ---modified and used this function:
# def remove_outliers(data, column, min_val, max_val):
#     """
#     Remove rows where column values are outside min_val and max_val.
#     """
#     return data[(data[column] > min_val) & (data[column] < max_val)].reset_index(drop=True)
# 
# def categorize_delays(data, column):
#     """
#     Categorize delays into Short, Medium, and Long.
#     """
#     data[column] = data[column].apply(
#         lambda x: "Short Delay" if x > 0 and x <= 10 else
#                   "Medium Delay" if x > 10 and x <= 20 else
#                   "Long Delay" if x > 20 else x
#     )
#     return data
# ---
# 
# def preprocess_and_split(data, target_column, numeric_features, categorical_features, test_size=0.2, random_state=123):
#     """
#     Preprocess data and split into train and test sets.
#     """
#     X = data.drop(columns=[target_column])
#     y = data[target_column]
# 
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', StandardScaler(), numeric_features),
#             ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
#         ]
#     )
# 
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
#     preprocessor.fit(X_train)
#     
#     train_processed = preprocessor.transform(X_train)
#     test_processed = preprocessor.transform(X_test)
#     return X_train, X_test, y_train, y_test, train_processed, test_processed, preprocessor
# 
# def save_artifacts(preprocessor, X_train, y_train, X_test, y_test, train_processed, test_processed, output_paths):
#     """
#     Save preprocessor and datasets to disk.
#     """
#     pickle.dump(preprocessor, open(output_paths['preprocessor'], 'wb'))
#     X_train.to_csv(output_paths['X_train'], index=False)
#     y_train.to_csv(output_paths['y_train'], index=False)
#     X_test.to_csv(output_paths['X_test'], index=False)
#     y_test.to_csv(output_paths['y_test'], index=False)
#     pickle.dump(train_processed, open(output_paths['train_processed'], 'wb'))
#     pickle.dump(test_processed, open(output_paths['test_processed'], 'wb'))
# """
