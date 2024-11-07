# Crypto AI Model

A comprehensive cryptocurrency AI model designed to detect trends, predict trend reversals, estimate peak prices, and execute automated trading strategies using Binance data and deep learning techniques.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
  - [Data Fetching](#data-fetching)
  - [Data Preprocessing](#data-preprocessing)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Model Training](#model-training)
  - [Backtesting](#backtesting)
  - [Real-Time Prediction and Trading](#real-time-prediction-and-trading)
  - [Monitoring](#monitoring)
- [Scalability](#scalability)
- [Logging](#logging)
- [Testing](#testing)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Data Collection:** Fetches historical and real-time cryptocurrency data from Binance.
- **Data Preprocessing:** Cleans data and engineers a rich set of technical indicators.
- **Deep Learning Model:** Utilizes LSTM networks for trend reversal prediction.
- **Hyperparameter Tuning:** Optimizes model performance using Optuna.
- **Class Imbalance Handling:** Balances data using resampling and class weights.
- **Backtesting:** Evaluates trading strategies on historical data.
- **Real-Time Prediction:** Makes live predictions on incoming data streams.
- **Automated Trading:** Executes buy/sell orders based on model predictions.
- **Monitoring:** Sends alerts and notifications via Telegram.
- **Scalability:** Containerized with Docker for easy deployment and scaling.

## Project Structure

crypto_ai_model/ ├── data/ │ ├── raw/ # Raw data fetched from Binance │ ├── processed/ # Processed data with features and labels │ └── external/ # External datasets (if any) ├── src/ │ ├── init.py │ ├── config.py # Configuration settings │ ├── data_fetcher.py # Module to fetch data from Binance │ ├── preprocessing.py # Data preprocessing and feature engineering │ ├── model.py # Model building and training │ ├── evaluation.py # Model evaluation and visualization │ ├── tuning.py # Hyperparameter tuning │ ├── imbalance.py # Handling class imbalance │ ├── real_time.py # Real-time data streaming and prediction │ ├── trading.py # Automated trading strategies │ ├── backtesting.py # Backtesting trading strategies │ ├── monitoring.py # Monitoring and alerts via Telegram │ └── utils.py # Utility functions ├── models/ # Saved models │ └── lstm_model.h5 ├── notebooks/ # Jupyter notebooks for exploration │ └── exploratory_analysis.ipynb ├── scripts/ # Scripts for running the pipeline │ ├── run_data_fetching.py │ ├── run_preprocessing.py │ ├── run_training.py │ ├── run_evaluation.py │ ├── run_tuning.py │ ├── run_real_time.py │ ├── run_trading.py │ ├── run_backtesting.py │ └── run_monitoring.py ├── docker/ │ ├── Dockerfile │ └── docker-compose.yml ├── logs/ # Log files │ ├── data_fetcher.log │ ├── trading.log │ ├── pipeline.log │ ├── backtesting.log │ └── monitoring.log ├── requirements.txt # Python dependencies ├── README.md # Project documentation └── .gitignore # Git ignore file



## Setup Instructions

### 1. Clone the Repository

git clone https://github.com/yourusername/crypto_ai_model.git
cd crypto_ai_model

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


pip install -r requirements.txt

### 2. Create a Virtual Environment

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

### 3. Install Dependencies

python3 -m pip install -r requirements.txt

### 4. Set Binance API Keys (Optional)

### 5. Run the pipeline

python src/scripts/run_pipeline.py

### 6. Building and Running the Docker Container
$ cd docker && docker-compose build
$ docker-compose up -d


### 7. Documentation
#### 7.1 Initialize Sphinx
sphinx-quickstart docs

#### 7.2 Build Documentation
cd docs
make html

#### 7.3 Access Documentation:
Open docs/_build/html/index.html in a web browser.


## 3. Usage
### 3.1 Data Fetching
$ python src/scripts/run_data_fetching.py
or 
$ python src/scripts/run_data_fetching.py --symbol BTCUSDT,ETHUSDT,XRPUSDT

### 3.2 Data Preprocessing
python src/scripts/run_preprocessing.py

### 3.3 Hyperparameter Tuning
python src/scripts/run_tuning.py

### 3.4 Model Training
python src/scripts/run_training.py

### 3.5 Backtesting
python src/scripts/run_backtesting.py

### 3.6 Real-Time Prediction and Trading
python src/scripts/run_real_time.py

### 3.7 Monitoring
python src/scripts/run_monitoring.py

### 3.8 Testing
python -m unittest discover -s tests


## 4. Docker
### 4.1 Running Docker
cd docker
docker-compose build
docker-compose up -d

## 5 Documentation
### 5.1 Initializing Sphinx in Your Project
sphinx-quickstart docs

### 5.2 Building the Documentation
cd docs && make html

### 5.3 View the Documentation
open build/html/index.html  # On macOS
xdg-open build/html/index.html  # On Linux
start build/html/index.html  # On Windows


### Running Scripts with Dynamic Symbols

You can specify the trading symbol dynamically using command-line arguments. If not specified, the default symbol set in the `.env` file will be used.

**Example: Fetching Data for BTCUSDT and ETHUSDT**

python src/scripts/run_data_fetching.py --symbol BTCUSDT,ETHUSDT

#### Available Arguments
--symbol: Trading pair symbol(s), separated by commas (e.g., BTCUSDT,ETHUSDT).
--interval: Time interval for data (default: 1h).
--lookback: Lookback period for historical data (default: 1 year ago UTC).




