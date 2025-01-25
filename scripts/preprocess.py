# essential imports for pre processing
import sys
import os
import click
import pickle
import numpy as np
import pandas as pd
import altair as alt
# Simplify working with large datasets in Altair
alt.data_transformers.enable('vegafusion')
import matplotlib.pyplot as plt
import pandera as pa
from sklearn import set_config
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import cross_validate, RandomizedSearchCV
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.split import split
from src.preprocessing import preprocess
import warnings
warnings.filterwarnings(action='ignore')

@click.command()
@click.option('--raw_data', type=str, help="Raw data location")
@click.option('--preprocessed_data', type=str, help="Folder where processed data is to be saved")
@click.option('--preprocessor_loc', type=str, help="Folder where the preprocessor object will be saved")
@click.option('--seed', type=int, help="Random seed, default value: 123", default=123)
def main(raw_data, preprocessed_data, preprocessor_loc, seed):
    #Make folders for placing processed files and models if they do not already exit
    os.makedirs(os.path.dirname(preprocessed_data), exist_ok=True)
    os.makedirs(os.path.dirname(preprocessor_loc), exist_ok=True)

    # the Dates column is parsed through date argument to make sure it is seen as datetime object
    ttc = pd.read_csv(raw_data, parse_dates=['Date'])

    print(f"This dataset has {ttc.shape[0]} rows and {ttc.shape[1]} columns")

    # Fix the 'Time' column
    ttc1 = ttc.copy() # preserving the original

    # Convert column to datetime object
    ttc1['Time'] = pd.to_datetime(ttc['Time']).dt.time

    ttc1['Date_'] = ttc1['Date'].dt.date
    ttc1['Month'] = ttc1['Date'].dt.month
    ttc1['Hour'] = ttc1['Time'].map(lambda x: x.hour) # used ChatGPT for this conversion

    # Drop the 'Date' and 'Time' column since they are no longer needed
    ttc1 = ttc1.drop(columns=['Date', 'Time'])

    # Drop 'Direction' and 'Vehicle' columns as they are not useful
    ttc_clean = ttc1.drop(columns=['Direction', 'Vehicle'])
    # As NaN routes have 0 delays, it is safe to drop these rows
    ttc_clean = ttc_clean.dropna()
    ttc_clean.to_csv(os.path.join(preprocessed_data,'complete_data.csv'),index=False)
    ttc_clean.reset_index(drop=True, inplace=True)
    #Remove outliers with delay greater than 30 mins and less than 1 min
    ttc_lr = ttc_clean.loc[(ttc_clean["Min Delay"]<30) & (ttc_clean["Min Delay"]>0)].reset_index(drop=True)
    
    # Store data without outliers
    ttc_lr.to_csv(os.path.join(preprocessed_data,'no_outliers.csv'),index=False)
    # Assign delays into 3 classes "Short Delay", "Medium Delay", and "Long Delay"
    ttc_lr["Min Delay"] = ttc_lr["Min Delay"].apply(lambda x: "Short Delay" if type(x)== int and x >0 and x<=10 else x)
    ttc_lr["Min Delay"] = ttc_lr["Min Delay"].apply(lambda x: "Medium Delay" if type(x)== int and  x >10 and x<=20 else x)
    ttc_lr["Min Delay"] = ttc_lr["Min Delay"].apply(lambda x: "Long Delay" if type(x)== int and  x >=10 else x)

    # Define feature types
    numeric_features=["Hour","Month"]
    categorical_features = ['Location', 'Route', 'Incident',"Day"]

    split(ttc_lr,'Min Delay',preprocessed_data)

    X_train=pd.read_csv(os.path.join(preprocessed_data,'X_train.csv'))
    X_test=pd.read_csv(os.path.join(preprocessed_data,'X_test.csv'))
    preprocess(X_train,X_test, numeric_features,categorical_features,preprocessor_loc)

    


if __name__ =='__main__':
    main()