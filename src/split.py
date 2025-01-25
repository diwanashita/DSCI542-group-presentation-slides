import pandas as pd
import os
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
def split(data:pd.DataFrame, target_column: str,directory: str, index: bool = False, test_size: float=0.2, random_state: int=123):
    """
    Split data into train and test sets and save them to disk
    
    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to save as csv.
    target_column : str
        The column which is chosen as the target
    directory : str
        The directory where the file is to be saved.
    index : bool, optional
        Whether to include the DataFrame's index in the CSV file. Default is False.
    test_size : float, optional
        Ratio of test set for splitting into training and test sets. default is 0.2
    random_state: int, optional
        random seed for setting random number generators, default 123

    Raises
    ------
    ValueError
        If the DataFrame is empty.
    FileNotFoundError
        If the specified directory does not exist.
    TypeError
        If the input is not a pandas DataFrame.
    """

    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist.")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")    
    if data.empty:
        raise ValueError("DataFrame must contain observations.")
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    X_train.to_csv(os.path.join(directory,'X_train.csv'),index=False)
    y_train.to_csv(os.path.join(directory,'y_train.csv'),index=False)
    X_test.to_csv(os.path.join(directory,'X_test.csv'),index=False)
    y_test.to_csv(os.path.join(directory,'y_test.csv'),index=False)
    print(f"Saved train and test data in {directory} ")
