import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import random
import requests
import retrying
from config import data_base_path
from sklearn.linear_model import LinearRegression



forecast_price = {}

# Đường dẫn chứa dữ liệu Binance (sẽ lưu tại data_base_path/binance/futures-klines)
binance_data_path = os.path.join(data_base_path, "binance/futures-klines")
MAX_DATA_SIZE = 100       # Giới hạn số lượng dữ liệu tối đa khi lưu trữ
INITIAL_FETCH_SIZE = 100  # Số lượng nến lần đầu tải về

@retrying.retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=5)
def fetch_prices(symbol, interval="1m", limit=100, start_time=None, end_time=None):
    """
    Lấy dữ liệu giá từ API Binance.
    """
    try:
        base_url = "https://fapi.binance.com"
        endpoint = "/fapi/v1/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        url = base_url + endpoint
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Failed to fetch prices for {symbol} from Binance API: {str(e)}")
        raise e

def download_data(token):
    """
    Tải dữ liệu giá của token từ API Binance và lưu vào file CSV.
    Chúng ta chỉ làm việc với token BTC.
    """
    # Với BTC, symbol sẽ là "BTCUSDT"
    symbols = f"{token.upper()}USDT"
    interval = "5m"
    current_datetime = datetime.now()
    download_path = os.path.join(binance_data_path, token.lower())
    file_path = os.path.join(download_path, f"{token.lower()}_5m_data.csv")

    if os.path.exists(file_path):
        # Nếu file đã tồn tại, tải 100 cây nến trong khoảng 500 phút gần đây
        start_time = int((current_datetime - timedelta(minutes=500)).timestamp() * 1000)
        end_time = int(current_datetime.timestamp() * 1000)
        new_data = fetch_prices(symbols, interval, 100, start_time, end_time)
    else:
        # Nếu file không tồn tại, tải INITIAL_FETCH_SIZE nến
        start_time = int((current_datetime - timedelta(minutes=INITIAL_FETCH_SIZE * 5)).timestamp() * 1000)
        end_time = int(current_datetime.timestamp() * 1000)
        new_data = fetch_prices(symbols, interval, INITIAL_FETCH_SIZE, start_time, end_time)

    new_df = pd.DataFrame(new_data, columns=[
        "start_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", 
        "taker_buy_quote_asset_volume", "ignore"
    ])

    # Kết hợp dữ liệu cũ và mới nếu file đã tồn tại
    if os.path.exists(file_path):
        old_df = pd.read_csv(file_path)
        combined_df = pd.concat([old_df, new_df])
        combined_df = combined_df.drop_duplicates(subset=['start_time'], keep='last')
    else:
        combined_df = new_df

    # Giới hạn số lượng dữ liệu tối đa
    if len(combined_df) > MAX_DATA_SIZE:
        combined_df = combined_df.iloc[-MAX_DATA_SIZE:]

    if not os.path.exists(download_path):
        os.makedirs(download_path)
    combined_df.to_csv(file_path, index=False)
    print(f"Updated data for {token} saved to {file_path}. Total rows: {len(combined_df)}")

def format_data(token):
    """
    Định dạng dữ liệu đã tải từ Binance và lưu ra file CSV chuẩn để dự báo.
    """
    path = os.path.join(binance_data_path, token.lower())
    file_path = os.path.join(path, f"{token.lower()}_5m_data.csv")

    if not os.path.exists(file_path):
        print(f"No data file found for {token}")
        return

    df = pd.read_csv(file_path)
    columns_to_use = [
        "start_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"
    ]

    if set(columns_to_use).issubset(df.columns):
        df = df[columns_to_use]
        df.columns = [
            "start_time", "open", "high", "low", "close", "volume",
            "end_time", "quote_asset_volume", "n_trades", 
            "taker_volume", "taker_volume_usd"
        ]
        df.index = pd.to_datetime(df["start_time"], unit='ms')
        df.index.name = "date"

        output_path = os.path.join(data_base_path, f"{token.lower()}_price_data.csv")
        df.sort_index().to_csv(output_path)
        print(f"Formatted data saved to {output_path}")
    else:
        print(f"Required columns are missing in {file_path}. Skipping this file.")

def train_model(token):
    """
    Huấn luyện mô hình dự báo giá cho token dựa theo cấu hình model_config.
    Hỗ trợ Linear Regression, LSTM và XGBoost.
    """
    time_start = datetime.now()

    file_path = os.path.join(data_base_path, f"{token.lower()}_price_data.csv")
    try:
        price_data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Cannot read file {file_path}: {str(e)}")
        return

    price_data["date"] = pd.to_datetime(price_data["date"])
    price_data.set_index("date", inplace=True)
    df = price_data.resample('20T').mean().dropna()

    if model_config["model_type"] == "LinearRegression":
        X = np.array(range(len(df))).reshape(-1, 1)
        y = df['close'].values
        model = LinearRegression()
        model.fit(X, y)
        next_time_index = np.array([[len(df)]])
        predicted_price = model.predict(next_time_index)[0]

    elif model_config["model_type"] == "LSTM":
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        from tensorflow.keras.optimizers import Adam

        close_prices = df['close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        close_scaled = scaler.fit_transform(close_prices)
        window_size = model_config["window_size"]

        X_seq, y_seq = [], []
        for i in range(len(close_scaled) - window_size):
            X_seq.append(close_scaled[i:i+window_size])
            y_seq.append(close_scaled[i+window_size])
        X_seq, y_seq = np.array(X_seq), np.array(y_seq)

        model = Sequential()
        model.add(LSTM(model_config["lstm_units"], input_shape=(window_size, 1)))
        model.add(Dense(1))
        optimizer = Adam(learning_rate=model_config["learning_rate"])
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        model.fit(X_seq, y_seq, epochs=model_config["epochs"], batch_size=model_config["batch_size"], verbose=1)
        last_sequence = close_scaled[-window_size:]
        last_sequence = np.expand_dims(last_sequence, axis=0)
        predicted_scaled = model.predict(last_sequence)
        predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]

    elif model_config["model_type"] == "XGBoost":
        from xgboost import XGBRegressor

        X = np.array(range(len(df))).reshape(-1, 1)
        y = df['close'].values
        xgb_params = model_config["xgb_params"]
        model = XGBRegressor(
            max_depth=xgb_params["max_depth"],
            n_estimators=xgb_params["n_estimators"],
            learning_rate=xgb_params["learning_rate"],
            objective=xgb_params["objective"]
        )
        model.fit(X, y)
        next_time_index = np.array([[len(df)]])
        predicted_price = model.predict(next_time_index)[0]

    else:
        print("Unsupported model type!")
        return

    # Xác định khoảng dao động ±0.1% quanh giá dự đoán
    fluctuation_range = 0.001 * predicted_price
    min_price = predicted_price - fluctuation_range
    max_price = predicted_price + fluctuation_range
    price_predict = random.uniform(min_price, max_price)
    forecast_price[token] = price_predict

    print(f"Predicted_price: {predicted_price}, Min_price: {min_price}, Max_price: {max_price}")
    print(f"Forecasted price for {token}: {forecast_price[token]}")
    time_end = datetime.now()
    print(f"Time elapsed forecast: {time_end - time_start}")

def update_data():
    """
    Thực hiện tải dữ liệu, định dạng và dự đoán giá cho BTC.
    """
    token = "BTC"
    download_data(token)
    format_data(token)
    train_model(token)

if __name__ == "__main__":
    update_data()
