import pandas as pd
import os
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

def preprocess(X_train: pd.DataFrame, X_test: pd.DataFrame, numeric_features: list, categorical_features: list, path: str):
    """
    Preprocess data and split into train and test sets.
    
    Parameters
    ----------
    X_train: pd.Dataframe 
        Features used to train the model
    X_test: pd.Dataframe 
        Features to test the model 
    numeric_features: list 
        List of all numeric features
    categorical_features: list 
        List of all categorical features
    path: str
        Path where pre processor will be written to
    Raises
    ------
    ValueError
        If the DataFrame is empty.
    FileNotFoundError
        If the specified directory does not exist.
    TypeError
        If the input is not a pandas DataFrame.
    """
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory {directory} does not exist.")
    if not (isinstance(X_train, pd.DataFrame)or isinstance(X_test, pd.DataFrame)):
        raise TypeError("Input must be a pandas DataFrame")    
    if X_train.empty or X_test.empty:
        raise ValueError("DataFrame must contain observations.")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    pickle.dump(preprocessor,open(os.path.join(path,'delay_preprocessor.pickle'),'wb'))
    preprocessor.fit(X_train)
    train_processed=preprocessor.transform(X_train)
    test_processed=preprocessor.transform(X_test)
    pickle.dump(train_processed,open(os.path.join(path,'train_processed.pickle'),'wb'))
    pickle.dump(test_processed,open(os.path.join(path,'test_processed.pickle'),'wb'))