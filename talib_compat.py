
import sys
import numpy as np
import pandas as pd


try:
    import talib as real_talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

if not HAS_TALIB:
    class TALibCompat:
        @staticmethod
        def RSI(prices, timeperiod=14):
            prices = pd.Series(prices)
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(timeperiod).mean()
            loss = -delta.where(delta < 0, 0).rolling(timeperiod).mean()
            rs = gain / (loss + 1e-10)
            return (100 - 100 / (1 + rs)).values

        @staticmethod
        def MACD(prices, fastperiod=12, slowperiod=26, signalperiod=9):
            prices = pd.Series(prices)
            ema_fast = prices.ewm(span=fastperiod).mean()
            ema_slow = prices.ewm(span=slowperiod).mean()
            macd = ema_fast - ema_slow
            signal = macd.ewm(span=signalperiod).mean()
            hist = macd - signal
            return macd.values, signal.values, hist.values

        @staticmethod
        def BBANDS(prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0):
            prices = pd.Series(prices)
            middle = prices.rolling(timeperiod).mean()
            std = prices.rolling(timeperiod).std()
            upper = middle + std * nbdevup
            lower = middle - std * nbdevdn
            return upper.values, middle.values, lower.values

        @staticmethod
        def ATR(high, low, close, timeperiod=14):
            high = pd.Series(high)
            low = pd.Series(low)
            close = pd.Series(close)
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return tr.rolling(timeperiod).mean().values

        @staticmethod
        def STOCH(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0):
            high = pd.Series(high)
            low = pd.Series(low)
            close = pd.Series(close)
            lowest = low.rolling(fastk_period).min()
            highest = high.rolling(fastk_period).max()
            k = 100 * (close - lowest) / (highest - lowest + 1e-10)
            k = k.rolling(slowk_period).mean()
            d = k.rolling(slowd_period).mean()
            return k.values, d.values

        @staticmethod
        def ROC(prices, timeperiod=10):
            prices = pd.Series(prices)
            return ((prices / prices.shift(timeperiod) - 1) * 100).values

        @staticmethod
        def OBV(close, volume):
            close = pd.Series(close)
            volume = pd.Series(volume)
            return (np.sign(close.diff()) * volume).fillna(0).cumsum().values

    sys.modules['talib'] = TALibCompat()
else:
    # 使用真正的 talib
    sys.modules['talib'] = real_talib
