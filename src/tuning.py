import optuna
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from src.config import Config
from src.preprocessing import Preprocessor
from src.imbalance import ImbalanceHandler

class HyperparameterTuner:
    """
    Class to perform hyperparameter tuning using Optuna.
    """
    def __init__(self, processed_data_path, time_steps=60):
        """
        Initializes the HyperparameterTuner.
        
        Parameters:
        - processed_data_path (str): Path to the processed data CSV file.
        - time_steps (int): Number of time steps for LSTM input.
        """
        self.processed_data_path = processed_data_path
        self.time_steps = time_steps
        self.preprocessor = Preprocessor(
            raw_data_path=Config.DATA_RAW_PATH,
            processed_data_path=self.processed_data_path
        )
        self.imbalance_handler = ImbalanceHandler(method='resample')  # or 'class_weight'
    
    def objective(self, trial):
        """
        Objective function for Optuna to optimize.
        
        Parameters:
        - trial (optuna.trial.Trial): Optuna trial object.
        
        Returns:
        - float: Validation accuracy to maximize.
        """
        # Load and preprocess data
        df = self.preprocessor.load_data()
        df = self.preprocessor.handle_missing_values(df)
        df = self.preprocessor.feature_engineering(df)
        df = self.preprocessor.define_labels(df)
        features = [
            'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 
            'MACD', 'MACD_Signal', 'MACD_Diff', 'RSI', 'Volume_Change',
            'Bollinger_High', 'Bollinger_Low', 'Bollinger_Width',
            'Stochastic_%K', 'Stochastic_%D',
            'ADX', 'ADX_Pos_Directional', 'ADX_Neg_Directional'
        ]
        df = self.preprocessor.scale_features(df, features)
        
        # Handle class imbalance
        df = self.imbalance_handler.resample_data(df, target='Reversal')
        
        X = df[features].values
        y = df['Reversal'].values
        
        # Create sequences
        X_sequences, y_sequences = self.create_sequences(X, y, self.time_steps)
        
        # Split into training and validation sets
        split = int(0.8 * len(X_sequences))
        X_train = X_sequences[:split]
        y_train = y_sequences[:split]
        X_val = X_sequences[split:]
        y_val = y_sequences[split:]
        
        # Define hyperparameters to tune
        num_units = trial.suggest_categorical('num_units', [50, 100, 150])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        epochs = trial.suggest_int('epochs', 10, 50)
        
        # Build model
        model = Sequential()
        model.add(LSTM(units=num_units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(rate=dropout_rate))
        model.add(LSTM(units=num_units, return_sequences=False))
        model.add(Dropout(rate=dropout_rate))
        model.add(Dense(units=25, activation='relu'))
        model.add(Dense(units=1, activation='sigmoid'))
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train model
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            verbose=0
        )
        
        # Evaluate model
        y_pred_prob = model.predict(X_val)
        y_pred = (y_pred_prob > 0.5).astype(int)
        accuracy = accuracy_score(y_val, y_pred)
        
        return accuracy
    
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
    
    def tune(self, n_trials=50):
        """
        Runs the hyperparameter tuning process.
        
        Parameters:
        - n_trials (int): Number of Optuna trials.
        
        Returns:
        - dict: Best hyperparameters found.
        """
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        print(f"Best trial: {study.best_trial.params}")
        return study.best_trial.params

def main():
    tuner = HyperparameterTuner(processed_data_path=Config.DATA_PROCESSED_PATH, time_steps=Config.TIME_STEPS)
    best_params = tuner.tune(n_trials=50)
    # Optionally, save best_params to a file or integrate with ModelTrainer

if __name__ == "__main__":
    main()
