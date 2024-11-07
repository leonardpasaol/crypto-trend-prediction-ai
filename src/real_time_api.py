# src/real_time_api.py

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from src.config import Config
from src.preprocessing import Preprocessor

app = Flask(__name__)

# Load model and scaler
model = tf.keras.models.load_model(Config.MODEL_PATH)
preprocessor = Preprocessor(
    raw_data_path=Config.DATA_RAW_PATH,
    processed_data_path=Config.DATA_PROCESSED_PATH
)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Expecting JSON data with OHLCV
    try:
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Preprocessing
        df = preprocessor.handle_missing_values(df)
        df = preprocessor.feature_engineering(df)
        features = [
            'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 
            'MACD', 'MACD_Signal', 'MACD_Diff', 'RSI', 'Volume_Change',
            'Bollinger_High', 'Bollinger_Low', 'Bollinger_Width',
            'Stochastic_%K', 'Stochastic_%D',
            'ADX', 'ADX_Pos_Directional', 'ADX_Neg_Directional'
        ]
        feature_data = df[features].tail(Config.TIME_STEPS)
        
        # Scale features
        scaled_features = preprocessor.scaler.transform(feature_data)
        
        # Reshape for LSTM
        input_sequence = np.expand_dims(scaled_features, axis=0)
        
        # Predict
        prediction_prob = model.predict(input_sequence)
        prediction = (prediction_prob > 0.5).astype(int)[0][0]
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(prediction_prob[0][0])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000, debug=True)
