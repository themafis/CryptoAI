import requests
import pandas as pd
import pandas_ta as ta
from fastapi import FastAPI
import math
import os
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API çalışıyor!"}

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
    # OHLCV: [timestamp, open, high, low, close]
    return {"ohlc": data}

@app.get("/rsi/{coin_id}")
def get_coin_rsi(coin_id: str = "bitcoin", days: int = 1):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days={days}"
    response = requests.get(url)
    data = response.json()
    if not data:
        return {"error": "Veri yok"}
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
    rsi = ta.rsi(df["close"], length=14)
    return {"rsi": rsi.dropna().tolist()}

@app.get("/indicators/{coin_id}")
def get_all_indicators(coin_id: str = "bitcoin", days: int = 30):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days={days}"
        response = requests.get(url)
        data = response.json()
        print(f"[DEBUG] API'den dönen ham veri: {data}")
        if not data or not isinstance(data, list) or len(data) == 0:
            print("[ERROR] Veri yok veya CoinGecko API boş döndü")
            return {"error": "Veri yok veya CoinGecko API boş döndü"}
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
        print(f"[DEBUG] Oluşan DataFrame: {df.head()}")
        if df.empty or df.isnull().all().all():
            print("[ERROR] DataFrame tamamen boş veya tüm değerler NaN")
            return {"error": "Veri DataFrame'e aktarılamadı veya tamamen boş"}
        # Sütunlarda None veya tümü NaN ise
        for col in ["open", "high", "low", "close"]:
            if col not in df.columns or df[col].isnull().all():
                print(f"[ERROR] {col} sütunu boş veya eksik!")
                return {"error": f"{col} sütunu boş veya eksik"}
        result = {}
        # Temel indikatörler
        result["RSI"] = ta.rsi(df["close"], length=14).dropna().tolist() if not df["close"].isnull().all() else []
        macd_df = ta.macd(df["close"]) if not df["close"].isnull().all() else pd.DataFrame()
        result["MACD"] = macd_df.iloc[-1].to_dict() if not macd_df.empty else {}
        result["EMA_12"] = ta.ema(df["close"], length=12).dropna().tolist() if not df["close"].isnull().all() else []
        result["EMA_26"] = ta.ema(df["close"], length=26).dropna().tolist() if not df["close"].isnull().all() else []
        result["SMA_20"] = ta.sma(df["close"], length=20).dropna().tolist() if not df["close"].isnull().all() else []
        adx_df = ta.adx(df["high"], df["low"], df["close"]) if not df["high"].isnull().all() and not df["low"].isnull().all() and not df["close"].isnull().all() else pd.DataFrame()
        result["ADX"] = adx_df.iloc[-1].to_dict() if not adx_df.empty else {}
        bb_df = ta.bbands(df["close"])
        result["BollingerBands"] = bb_df.iloc[-1].to_dict() if not bb_df.empty else {}
        stochrsi_df = ta.stochrsi(df["close"])
        result["StochRSI"] = stochrsi_df.iloc[-1].to_dict() if not stochrsi_df.empty else {}
        result["CCI"] = ta.cci(df["high"], df["low"], df["close"]).dropna().tolist() if not df["close"].isnull().all() else []
        result["ATR"] = ta.atr(df["high"], df["low"], df["close"]).dropna().tolist() if not df["close"].isnull().all() else []
        result["OBV"] = ta.obv(df["close"], df["close"]).dropna().tolist() if not df["close"].isnull().all() else []
        if "volume" in df.columns and not df["volume"].isnull().all():
            result["VWAP"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"]).dropna().tolist()
        else:
            result["VWAP"] = None
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
        # Ichimoku hesaplama (tuple unpack ile, güvenli)
        try:
            ichimoku = ta.ichimoku(df["high"], df["low"], df["close"])
            if isinstance(ichimoku, tuple):
                conversion, base, span_a, span_b = ichimoku
                if all([hasattr(x, 'iloc') and not x.empty for x in [conversion, base, span_a, span_b]]):
                    result["Ichimoku"] = {
                        "conversion": conversion.iloc[-1],
                        "base": base.iloc[-1],
                        "span_a": span_a.iloc[-1],
                        "span_b": span_b.iloc[-1]
                    }
                else:
                    result["Ichimoku"] = {}
            else:
                result["Ichimoku"] = {}
        except Exception:
            result["Ichimoku"] = {}
        psar_df = ta.psar(df["high"], df["low"])
        result["ParabolicSAR"] = psar_df.iloc[-1].to_dict() if not psar_df.empty else {}
        result["WilliamsR"] = ta.willr(df["high"], df["low"], df["close"]).dropna().tolist() if not df["close"].isnull().all() else []
        supertrend_df = ta.supertrend(df["high"], df["low"], df["close"])
        result["Supertrend"] = supertrend_df.iloc[-1].to_dict() if not supertrend_df.empty else {}
        result["MFI"] = ta.mfi(df["high"], df["low"], df["close"], df["close"]).dropna().tolist() if not df["close"].isnull().all() else []
        donchian_df = ta.donchian(df["high"], df["low"])
        result["Donchian"] = donchian_df.iloc[-1].to_dict() if not donchian_df.empty else {}
        keltner_df = ta.kc(df["high"], df["low"], df["close"])
        result["Keltner"] = keltner_df.iloc[-1].to_dict() if not keltner_df.empty else {}
        result["Volume"] = df["close"].tolist() if not df["close"].isnull().all() else []
        print(f"[DEBUG] Hesaplanan indikatörler: {result}")
        return clean_json(result)
    except Exception as e:
        print(f"[EXCEPTION] {str(e)}")
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
    # open_time'ı datetime'a çevir ve index yap
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
        # Hacim gerektirmeyenler
        result["RSI"] = ta.rsi(df["close"], length=14).dropna().tolist()
        result["MACD"] = ta.macd(df["close"]).iloc[-1].to_dict() if not ta.macd(df["close"]).empty else {}
        result["EMA_12"] = ta.ema(df["close"], length=12).dropna().tolist()
        result["EMA_26"] = ta.ema(df["close"], length=26).dropna().tolist()
        result["SMA_20"] = ta.sma(df["close"], length=20).dropna().tolist()
        result["ADX"] = ta.adx(df["high"], df["low"], df["close"]).iloc[-1].to_dict() if not ta.adx(df["high"], df["low"], df["close"]).empty else {}
        result["BollingerBands"] = ta.bbands(df["close"]).iloc[-1].to_dict() if not ta.bbands(df["close"]).empty else {}
        result["StochRSI"] = ta.stochrsi(df["close"]).iloc[-1].to_dict() if not ta.stochrsi(df["close"]).empty else {}
        result["CCI"] = ta.cci(df["high"], df["low"], df["close"]).dropna().tolist()
        result["ATR"] = ta.atr(df["high"], df["low"], df["close"]).dropna().tolist()
        # Klasik pivot noktası hesaplama (manuel)
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
        # Ichimoku hesaplama (tuple unpack ile, güvenli)
        try:
            ichimoku = ta.ichimoku(df["high"], df["low"], df["close"])
            if isinstance(ichimoku, tuple):
                conversion, base, span_a, span_b = ichimoku
                if all([hasattr(x, 'iloc') and not x.empty for x in [conversion, base, span_a, span_b]]):
                    result["Ichimoku"] = {
                        "conversion": conversion.iloc[-1],
                        "base": base.iloc[-1],
                        "span_a": span_a.iloc[-1],
                        "span_b": span_b.iloc[-1]
                    }
                else:
                    result["Ichimoku"] = {}
            else:
                result["Ichimoku"] = {}
        except Exception:
            result["Ichimoku"] = {}
        result["ParabolicSAR"] = ta.psar(df["high"], df["low"]).iloc[-1].to_dict() if not ta.psar(df["high"], df["low"]).empty else {}
        result["WilliamsR"] = ta.willr(df["high"], df["low"], df["close"]).dropna().tolist()
        result["Donchian"] = ta.donchian(df["high"], df["low"]).iloc[-1].to_dict() if not ta.donchian(df["high"], df["low"]).empty else {}
        result["Keltner"] = ta.kc(df["high"], df["low"], df["close"]).iloc[-1].to_dict() if not ta.kc(df["high"], df["low"], df["close"]).empty else {}
        # Hacim gerektirenler (sadece Binance ile)
        result["VWAP"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"]).dropna().tolist()
        result["MFI"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"]).dropna().tolist()
        result["OBV"] = ta.obv(df["close"], df["volume"]).dropna().tolist()
        # Ekstra hacim
        result["Volume"] = df["volume"].tolist()
        return clean_json(result)
    except Exception as e:
        return {"error": str(e)}

@app.get("/indicators_coingecko/{coin_id}")
def get_all_indicators_coingecko(coin_id: str = "bitcoin", days: int = 30):
    try:
        df = get_coingecko_ohlc(coin_id, days)
        result = {}
        result["RSI"] = ta.rsi(df["close"], length=14).dropna().tolist()
        result["MACD"] = ta.macd(df["close"]).iloc[-1].to_dict() if not ta.macd(df["close"]).empty else {}
        result["EMA_12"] = ta.ema(df["close"], length=12).dropna().tolist()
        result["EMA_26"] = ta.ema(df["close"], length=26).dropna().tolist()
        result["SMA_20"] = ta.sma(df["close"], length=20).dropna().tolist()
        result["ADX"] = ta.adx(df["high"], df["low"], df["close"]).iloc[-1].to_dict() if not ta.adx(df["high"], df["low"], df["close"]).empty else {}
        result["BollingerBands"] = ta.bbands(df["close"]).iloc[-1].to_dict() if not ta.bbands(df["close"]).empty else {}
        result["StochRSI"] = ta.stochrsi(df["close"]).iloc[-1].to_dict() if not ta.stochrsi(df["close"]).empty else {}
        result["CCI"] = ta.cci(df["high"], df["low"], df["close"]).dropna().tolist()
        result["ATR"] = ta.atr(df["high"], df["low"], df["close"]).dropna().tolist()
        # Klasik pivot noktası hesaplama (manuel)
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
        # Ichimoku hesaplama (tuple unpack ile, güvenli)
        try:
            ichimoku = ta.ichimoku(df["high"], df["low"], df["close"])
            if isinstance(ichimoku, tuple):
                conversion, base, span_a, span_b = ichimoku
                if all([hasattr(x, 'iloc') and not x.empty for x in [conversion, base, span_a, span_b]]):
                    result["Ichimoku"] = {
                        "conversion": conversion.iloc[-1],
                        "base": base.iloc[-1],
                        "span_a": span_a.iloc[-1],
                        "span_b": span_b.iloc[-1]
                    }
                else:
                    result["Ichimoku"] = {}
            else:
                result["Ichimoku"] = {}
        except Exception:
            result["Ichimoku"] = {}
        result["ParabolicSAR"] = ta.psar(df["high"], df["low"]).iloc[-1].to_dict() if not ta.psar(df["high"], df["low"]).empty else {}
        result["WilliamsR"] = ta.willr(df["high"], df["low"], df["close"]).dropna().tolist()
        result["Donchian"] = ta.donchian(df["high"], df["low"]).iloc[-1].to_dict() if not ta.donchian(df["high"], df["low"]).empty else {}
        result["Keltner"] = ta.kc(df["high"], df["low"], df["close"]).iloc[-1].to_dict() if not ta.kc(df["high"], df["low"], df["close"]).empty else {}
        return clean_json(result)
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
    # Detaylı bilgi çek
    details = requests.get(f"https://api.coingecko.com/api/v3/coins/{coin['id']}").json()
    return {
        "id": coin["id"],
        "symbol": coin["symbol"],
        "name": coin["name"],
        "image": details.get("image", {}).get("large"),
        "thumb": details.get("image", {}).get("thumb"),
        "small": details.get("image", {}).get("small"),
    }

def clean_json(obj):
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    else:
        return obj
    
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
