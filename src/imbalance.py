from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight
import numpy as np
import pandas as pd

class ImbalanceHandler:
    """
    Class to handle class imbalance using resampling or class weights.
    """
    def __init__(self, method='resample'):
        """
        Initializes the ImbalanceHandler.
        
        Parameters:
        - method (str): 'resample' or 'class_weight'.
        """
        self.method = method
        self.smote = SMOTE(random_state=42)
    
    def resample_data(self, df, target='Reversal'):
        """
        Applies SMOTE resampling to balance classes.
        
        Parameters:
        - df (pd.DataFrame): DataFrame containing features and target.
        - target (str): Name of the target column.
        
        Returns:
        - pd.DataFrame: Resampled DataFrame.
        """
        X = df.drop(columns=[target])
        y = df[target]
        X_res, y_res = self.smote.fit_resample(X, y)
        df_res = pd.DataFrame(X_res, columns=X.columns)
        df_res[target] = y_res
        return df_res
    
    def compute_class_weights(self, y):
        """
        Computes class weights for imbalanced classes.
        
        Parameters:
        - y (np.array): Array of target labels.
        
        Returns:
        - dict: Dictionary mapping class labels to weights.
        """
        classes = np.unique(y)
        weights = class_weight.compute_class_weight('balanced', classes=classes, y=y)
        class_weights = dict(zip(classes, weights))
        return class_weights
