import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from src.config import Config

class Evaluator:
    """
    Class to evaluate and visualize the trained model's performance.
    """
    def __init__(self, processed_data_path, model_path, time_steps=60):
        """
        Initializes the Evaluator.
        
        Parameters:
        - processed_data_path (str): Path to the processed data CSV file.
        - model_path (str): Path to the trained model.
        - time_steps (int): Number of time steps for LSTM input.
        """
        self.processed_data_path = processed_data_path
        self.model_path = model_path
        self.time_steps = time_steps
        self.model = load_model(self.model_path)
        self.symbol = Config.SYMBOL
    
    def load_processed_data(self):
        """
        Loads processed data from the CSV file.
        
        Returns:
        - pd.DataFrame: DataFrame containing processed data.
        """
        df = pd.read_csv(self.processed_data_path, parse_dates=['Date'])
        df.set_index('Date', inplace=True)
        return df
    
    def create_sequences(self, X, y, time_steps=60):
        """
        Creates sequences of data for LSTM input.
        
        Parameters:
        - X (np.array): Feature array.
        - y (np.array): Target array.
        - time_steps (int): Number of time steps.
        
        Returns:
        - Xs (np.array): Array of input sequences.
        - ys (np.array): Array of target values.
        """
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            Xs.append(X[i:(i + time_steps)])
            ys.append(y[i + time_steps])
        return np.array(Xs), np.array(ys)
    
    def prepare_data(self, df):
        """
        Prepares data for evaluation.
        
        Parameters:
        - df (pd.DataFrame): DataFrame containing features and target.
        
        Returns:
        - X_test (np.array): Test feature sequences.
        - y_test (np.array): Test target values.
        """
        features = [
            'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 
            'MACD', 'MACD_Signal', 'MACD_Diff', 'RSI', 'Volume_Change',
            'Bollinger_High', 'Bollinger_Low', 'Bollinger_Width',
            'Stochastic_%K', 'Stochastic_%D',
            'ADX', 'ADX_Pos_Directional', 'ADX_Neg_Directional'
        ]
        target = 'Reversal'
        X = df[features].values
        y = df[target].values
        
        # Create sequences
        X_sequences, y_sequences = self.create_sequences(X, y, self.time_steps)
        
        # Split into training and testing sets
        split = int(0.8 * len(X_sequences))
        X_test = X_sequences[split:]
        y_test = y_sequences[split:]
        
        return X_test, y_test
    
    def evaluate(self, X_test, y_test):
        """
        Evaluates the model's performance on the test set.
        
        Parameters:
        - X_test (np.array): Test feature sequences.
        - y_test (np.array): Test target values.
        
        Returns:
        - y_pred (np.array): Predicted target values.
        """
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Reversal', 'Reversal'], 
                    yticklabels=['No Reversal', 'Reversal'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.show()
        
        return y_pred
    
    def visualize_predictions(self, df, y_pred):
        """
        Visualizes predictions alongside actual close prices.
        
        Parameters:
        - df (pd.DataFrame): DataFrame containing features and target.
        - y_pred (np.array): Predicted target values.
        """
        # Align predictions with test set
        X_test, y_test = self.prepare_data(df)
        df_test = df.iloc[-len(y_pred):].copy()
        df_test['Reversal_Prediction'] = y_pred.flatten()
        df_test['Reversal_Actual'] = y_test

        plt.figure(figsize=(14,7))
        plt.plot(df_test.index, df_test['Close'], label='Close Price')
        reversals = df_test[df_test['Reversal_Prediction'] == 1]
        plt.scatter(reversals.index, reversals['Close'], marker='^', color='g', label='Predicted Reversal')
        plt.title(f'{self.symbol} Price with Predicted Reversals', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Price (USDT)', fontsize=14)
        plt.legend()
        plt.show()

    def run(self):
        """
        Executes the evaluation pipeline.
        """
        df = self.load_processed_data()
        X_test, y_test = self.prepare_data(df)
        y_pred = self.evaluate(X_test, y_test)
        self.visualize_predictions(df, y_pred)
