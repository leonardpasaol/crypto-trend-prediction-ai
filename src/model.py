import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from src.config import Config
from src.imbalance import ImbalanceHandler

class ModelTrainer:
    """
    Class to build, train, and save the LSTM model.
    """
    def __init__(self, processed_data_path, model_path, time_steps=60, imbalance_method='resample'):
        """
        Initializes the ModelTrainer.
        
        Parameters:
        - processed_data_path (str): Path to the processed data CSV file.
        - model_path (str): Path to save the trained model.
        - time_steps (int): Number of time steps for LSTM input.
        - imbalance_method (str): 'resample' or 'class_weight'.
        """
        self.processed_data_path = processed_data_path
        self.model_path = model_path
        self.time_steps = time_steps
        self.model = None
        self.imbalance_handler = ImbalanceHandler(method=imbalance_method)
    
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
        - y (pd.Series or np.array): Target array.
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
        Prepares data for model training.
        
        Parameters:
        - df (pd.DataFrame): DataFrame containing features and target.
        
        Returns:
        - X_train (np.array): Training feature sequences.
        - X_val (np.array): Validation feature sequences.
        - y_train (np.array): Training target values.
        - y_val (np.array): Validation target values.
        - class_weights (dict or None): Class weights if method is 'class_weight'.
        """
        features = [
            'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 
            'MACD', 'MACD_Signal', 'MACD_Diff', 'RSI', 'Volume_Change',
            'Bollinger_High', 'Bollinger_Low', 'Bollinger_Width',
            'Stochastic_%K', 'Stochastic_%D',
            'ADX', 'ADX_Pos_Directional', 'ADX_Neg_Directional'
        ]
        target = 'Reversal'
        X = df[features]
        y = df[target].values
        
        # Handle class imbalance
        if self.imbalance_handler.method == 'resample':
            df_resampled = self.imbalance_handler.resample_data(df, target=target)
            X = df_resampled[features].values
            y = df_resampled[target].values
            class_weights = None
        elif self.imbalance_handler.method == 'class_weight':
            class_weights = self.imbalance_handler.compute_class_weights(y)
        else:
            class_weights = None
        
        # Create sequences
        X_sequences, y_sequences = self.create_sequences(X, y, self.time_steps)
        
        # Split into training and validation sets
        split = int(0.8 * len(X_sequences))
        X_train = X_sequences[:split]
        y_train = y_sequences[:split]
        X_val = X_sequences[split:]
        y_val = y_sequences[split:]
        
        return X_train, X_val, y_train, y_val, class_weights
    
    def build_model(self, input_shape, num_units=50, dropout_rate=0.2):
        """
        Builds the LSTM model.
        
        Parameters:
        - input_shape (tuple): Shape of the input data.
        - num_units (int): Number of LSTM units.
        - dropout_rate (float): Dropout rate.
        
        Returns:
        - model (tf.keras.Model): Compiled LSTM model.
        """
        model = Sequential()
        model.add(LSTM(units=num_units, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(rate=dropout_rate))
        model.add(LSTM(units=num_units, return_sequences=False))
        model.add(Dropout(rate=dropout_rate))
        model.add(Dense(units=25, activation='relu'))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        return model
    
    def train(self, X_train, y_train, X_val, y_val, class_weights=None):
        """
        Trains the LSTM model.
        
        Parameters:
        - X_train (np.array): Training feature sequences.
        - y_train (np.array): Training target values.
        - X_val (np.array): Validation feature sequences.
        - y_val (np.array): Validation target values.
        - class_weights (dict or None): Class weights for training.
        
        Returns:
        - history (tf.keras.callbacks.History): Training history.
        """
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_model(input_shape)
        
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        if class_weights:
            history = self.model.fit(
                X_train, y_train, 
                epochs=Config.EPOCHS, 
                batch_size=Config.BATCH_SIZE, 
                validation_data=(X_val, y_val),
                class_weight=class_weights,
                callbacks=[early_stop]
            )
        else:
            history = self.model.fit(
                X_train, y_train, 
                epochs=Config.EPOCHS, 
                batch_size=Config.BATCH_SIZE, 
                validation_data=(X_val, y_val),
                callbacks=[early_stop]
            )
        return history
    
    def save_model(self):
        """
        Saves the trained model to the specified path.
        """
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def run(self):
        """
        Executes the entire model training pipeline.
        """
        df = self.load_processed_data()
        X_train, X_val, y_train, y_val, class_weights = self.prepare_data(df)
        history = self.train(X_train, y_train, X_val, y_val, class_weights)
        self.save_model()
        # Optionally, save training history or additional metrics
