import requests
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import math
import os
import uvicorn

app = FastAPI()

# CORS ayarları (iOS app'inizden erişim için)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production'da spesifik domain'ler belirtin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "CryptoAI Backend API çalışıyor!"}

# Manuel RSI hesaplama
def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gains = pd.Series(gains).rolling(window=period).mean()
    avg_losses = pd.Series(losses).rolling(window=period).mean()
    
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Manuel MACD hesaplama
def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = pd.Series(prices).ewm(span=fast).mean()
    ema_slow = pd.Series(prices).ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# Manuel EMA hesaplama
def calculate_ema(prices, period):
    return pd.Series(prices).ewm(span=period).mean()

# Manuel SMA hesaplama
def calculate_sma(prices, period):
    return pd.Series(prices).rolling(window=period).mean()

# Manuel Bollinger Bands hesaplama
def calculate_bollinger_bands(prices, period=20, std_dev=2):
    sma = calculate_sma(prices, period)
    std = pd.Series(prices).rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

@app.get("/price/{coin_id}")
def get_coin_price(coin_id: str = "bitcoin"):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
    response = requests.get(url)
    data = response.json()
    return {coin_id: data.get(coin_id, {})}

@app.get("/ohlc/{coin_id}")
def get_coin_ohlc(coin_id: str = "bitcoin", days: int = 1):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days={days}"
    response = requests.get(url)
    data = response.json()
    return {"ohlc": data}

@app.get("/rsi/{coin_id}")
def get_coin_rsi(coin_id: str = "bitcoin", days: int = 1):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days={days}"
    response = requests.get(url)
    data = response.json()
    if not data:
        return {"error": "Veri yok"}
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
    rsi = calculate_rsi(df["close"].values, 14)
    return {"rsi": rsi.dropna().tolist()}

@app.get("/indicators/{coin_id}")
def get_all_indicators(coin_id: str = "bitcoin", days: int = 30):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days={days}"
        response = requests.get(url)
        data = response.json()
        
        if not data or not isinstance(data, list) or len(data) == 0:
            return {"error": "Veri yok veya CoinGecko API boş döndü"}
        
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
        
        if df.empty or df.isnull().all().all():
            return {"error": "Veri DataFrame'e aktarılamadı veya tamamen boş"}
        
        result = {}
        prices = df["close"].values
        
        # Manuel hesaplamalar
        result["RSI"] = calculate_rsi(prices, 14).dropna().tolist()
        
        macd_line, signal_line, histogram = calculate_macd(prices)
        result["MACD"] = {
            "MACD_12_26_9": macd_line.iloc[-1] if not macd_line.empty else None,
            "MACDs_12_26_9": signal_line.iloc[-1] if not signal_line.empty else None,
            "MACDh_12_26_9": histogram.iloc[-1] if not histogram.empty else None
        }
        
        result["EMA_12"] = calculate_ema(prices, 12).dropna().tolist()
        result["EMA_26"] = calculate_ema(prices, 26).dropna().tolist()
        result["SMA_20"] = calculate_sma(prices, 20).dropna().tolist()
        
        # Bollinger Bands
        upper, middle, lower = calculate_bollinger_bands(prices)
        result["BollingerBands"] = {
            "BBU_20_2.0": upper.iloc[-1] if not upper.empty else None,
            "BBM_20_2.0": middle.iloc[-1] if not middle.empty else None,
            "BBL_20_2.0": lower.iloc[-1] if not lower.empty else None
        }
        
        # Pivot Points
        pivot = ((df["high"] + df["low"] + df["close"]) / 3)
        r1 = (2 * pivot) - df["low"]
        s1 = (2 * pivot) - df["high"]
        r2 = pivot + (df["high"] - df["low"])
        s2 = pivot - (df["high"] - df["low"])
        result["PivotPoints"] = {
            "pivot": pivot.iloc[-1] if not pivot.empty else None,
            "resistance1": r1.iloc[-1] if not r1.empty else None,
            "support1": s1.iloc[-1] if not s1.empty else None,
            "resistance2": r2.iloc[-1] if not r2.empty else None,
            "support2": s2.iloc[-1] if not s2.empty else None
        }
        
        # Basit değerler
        result["Volume"] = df["close"].tolist()  # Volume yoksa close kullan
        result["ADX"] = {"ADX_14": 25.0}  # Basit değer
        
        return result
    except Exception as e:
        print(f"[ERROR] Hata: {e}")
        return {"error": str(e)}

def get_binance_ohlcv(symbol="BTCUSDT", interval="1d", limit=30):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    return df

def get_coingecko_ohlc(coin_id="bitcoin", days=30):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days={days}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    return df

@app.get("/indicators_binance/{symbol}")
def get_all_indicators_binance(symbol: str = "BTCUSDT", interval: str = "1d", limit: int = 30):
    try:
        df = get_binance_ohlcv(symbol, interval, limit)
        result = {}
        prices = df["close"].values
        
        # Manuel hesaplamalar
        result["RSI"] = calculate_rsi(prices, 14).dropna().tolist()
        
        macd_line, signal_line, histogram = calculate_macd(prices)
        result["MACD"] = {
            "MACD_12_26_9": macd_line.iloc[-1] if not macd_line.empty else None,
            "MACDs_12_26_9": signal_line.iloc[-1] if not signal_line.empty else None,
            "MACDh_12_26_9": histogram.iloc[-1] if not histogram.empty else None
        }
        
        result["EMA_12"] = calculate_ema(prices, 12).dropna().tolist()
        result["EMA_26"] = calculate_ema(prices, 26).dropna().tolist()
        result["SMA_20"] = calculate_sma(prices, 20).dropna().tolist()
        
        # Bollinger Bands
        upper, middle, lower = calculate_bollinger_bands(prices)
        result["BollingerBands"] = {
            "BBU_20_2.0": upper.iloc[-1] if not upper.empty else None,
            "BBM_20_2.0": middle.iloc[-1] if not middle.empty else None,
            "BBL_20_2.0": lower.iloc[-1] if not lower.empty else None
        }
        
        # Pivot Points
        pivot = (df["high"] + df["low"] + df["close"]) / 3
        r1 = (2 * pivot) - df["low"]
        s1 = (2 * pivot) - df["high"]
        r2 = pivot + (df["high"] - df["low"])
        s2 = pivot - (df["high"] - df["low"])
        result["PivotPoints"] = {
            "pivot": pivot.iloc[-1] if not pivot.empty else None,
            "resistance1": r1.iloc[-1] if not r1.empty else None,
            "support1": s1.iloc[-1] if not s1.empty else None,
            "resistance2": r2.iloc[-1] if not r2.empty else None,
            "support2": s2.iloc[-1] if not s2.empty else None
        }
        
        # Basit değerler
        result["Volume"] = df["volume"].tolist()
        result["ADX"] = {"ADX_14": 25.0}
        
        return result
    except Exception as e:
        return {"error": str(e)}

@app.get("/indicators_coingecko/{coin_id}")
def get_all_indicators_coingecko(coin_id: str = "bitcoin", days: int = 30):
    try:
        df = get_coingecko_ohlc(coin_id, days)
        result = {}
        prices = df["close"].values
        
        # Manuel hesaplamalar
        result["RSI"] = calculate_rsi(prices, 14).dropna().tolist()
        
        macd_line, signal_line, histogram = calculate_macd(prices)
        result["MACD"] = {
            "MACD_12_26_9": macd_line.iloc[-1] if not macd_line.empty else None,
            "MACDs_12_26_9": signal_line.iloc[-1] if not signal_line.empty else None,
            "MACDh_12_26_9": histogram.iloc[-1] if not histogram.empty else None
        }
        
        result["EMA_12"] = calculate_ema(prices, 12).dropna().tolist()
        result["EMA_26"] = calculate_ema(prices, 26).dropna().tolist()
        result["SMA_20"] = calculate_sma(prices, 20).dropna().tolist()
        
        # Bollinger Bands
        upper, middle, lower = calculate_bollinger_bands(prices)
        result["BollingerBands"] = {
            "BBU_20_2.0": upper.iloc[-1] if not upper.empty else None,
            "BBM_20_2.0": middle.iloc[-1] if not middle.empty else None,
            "BBL_20_2.0": lower.iloc[-1] if not lower.empty else None
        }
        
        # Pivot Points
        pivot = (df["high"] + df["low"] + df["close"]) / 3
        r1 = (2 * pivot) - df["low"]
        s1 = (2 * pivot) - df["high"]
        r2 = pivot + (df["high"] - df["low"])
        s2 = pivot - (df["high"] - df["low"])
        result["PivotPoints"] = {
            "pivot": pivot.iloc[-1] if not pivot.empty else None,
            "resistance1": r1.iloc[-1] if not r1.empty else None,
            "support1": s1.iloc[-1] if not s1.empty else None,
            "resistance2": r2.iloc[-1] if not r2.empty else None,
            "support2": s2.iloc[-1] if not s2.empty else None
        }
        
        # Basit değerler
        result["Volume"] = df["close"].tolist()
        result["ADX"] = {"ADX_14": 25.0}
        
        return result
    except Exception as e:
        return {"error": str(e)}

@app.get("/coin_info/{symbol}")
def get_coin_info(symbol: str):
    # Tüm coin listesini çek
    coins = requests.get("https://api.coingecko.com/api/v3/coins/list").json()
    # Sembol eşleşmesi bul
    coin = next((c for c in coins if c["symbol"].lower() == symbol.lower()), None)
    if not coin:
        return {"error": "Coin bulunamadı"}
    
    # Coin detaylarını çek
    coin_detail = requests.get(f"https://api.coingecko.com/api/v3/coins/{coin['id']}").json()
    
    return {
        "id": coin["id"],
        "symbol": coin["symbol"],
        "name": coin["name"],
        "image": coin_detail.get("image", {}).get("large"),
        "thumb": coin_detail.get("image", {}).get("thumb"),
        "small": coin_detail.get("image", {}).get("small")
    }

def clean_json(obj):
    """JSON serialization için temizleme"""
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json(item) for item in obj]
    elif isinstance(obj, (int, float)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    else:
        return obj

# Render için server başlatma
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
