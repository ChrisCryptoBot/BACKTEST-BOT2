import numpy as np
import sys
import os
import random
import json
import time
import logging
import pandas as pd
import pytz
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import requests
import threading
import pandas_ta as ta
import websocket
import asyncio
import uuid
from abc import ABC, abstractmethod
from collections import deque, defaultdict
from scipy.stats import norm
import lightgbm as lgb
from numba import jit

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('kraken_crypto_logger.log'), logging.StreamHandler()]
)
logger = logging.getLogger('kraken_crypto_logger')

# Kraken API credentials - Replace with your actual keys
KRAKEN_API_KEY = 'DW2lSFIA6qmHWIWy8fLz7CRn+kqOViETVkX0PIIJu4t8Rxg6Il7d+/IX'
KRAKEN_PRIVATE_KEY = 'minbBj7EH3W94UocrferA14SyIZ+aEKFs73b9xKx4vQNGM431SrV2VZM6S1lMNvfi3wQj6Vmge3m7kib80ARYToA'
GOOGLE_CREDENTIALS_FILE = os.path.expanduser("~/Downloads/credentials.json")

# TimeFrame class
class TimeFrame:
    Minute = 1
    Hour = 60
    Day = 1440

# Abstract base classes
class ExchangeClient(ABC):
    @abstractmethod
    def fetch_bars(self, symbol, timeframe, limit):
        pass
    
    @abstractmethod
    def fetch_trades(self, symbol, limit):
        pass
    
    @abstractmethod
    def get_order_book(self, symbol):
        pass

class TradingStrategy(ABC):
    @abstractmethod
    def analyze(self, symbol, exchange_client, **kwargs):
        pass
    
    @abstractmethod
    def get_metrics(self):
        pass
    
    def record_trade(self, trade_result):
        pass

# KrakenClient implementation
class KrakenClient(ExchangeClient):
    SYMBOL_MAPPING = {'BTC/USD': 'XXBTZUSD', 'ETH/USD': 'XETHZUSD', 'SOL/USD': 'SOLUSD'}

    def __init__(self, api_key, private_key):
        self.api_key = api_key
        self.private_key = private_key
        self.session = requests.Session()
        self.cache = {}
        self.cache_ttl = {"bars": {"1m": 60, "1h": 300, "1d": 3600}, "order_book": 30, "trades": 30}

    def fetch_bars(self, symbol, timeframe=TimeFrame.Minute, limit=50):
        cache_key = f"{symbol}_{timeframe}_{limit}"
        tf_key = "1m" if timeframe == TimeFrame.Minute else "1h" if timeframe == TimeFrame.Hour else "1d"
        if cache_key in self.cache and (time.time() - self.cache[cache_key]['timestamp']) < self.cache_ttl["bars"][tf_key]:
            return self.cache[cache_key]['data']
        
        kraken_symbol = self.SYMBOL_MAPPING.get(symbol, symbol)
        interval = {TimeFrame.Minute: 1, TimeFrame.Hour: 60, TimeFrame.Day: 1440}.get(timeframe)
        url = f"https://api.kraken.com/0/public/OHLC?pair={kraken_symbol}&interval={interval}"
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if 'error' in data and data['error']:
                logger.error(f"Kraken API error: {data['error']}")
                return pd.DataFrame()
            bars = data['result'].get(kraken_symbol)
            if not bars:
                return pd.DataFrame()
            df = pd.DataFrame(bars, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            result = df.tail(limit)
            self.cache[cache_key] = {'data': result, 'timestamp': time.time()}
            return result
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return pd.DataFrame()

    def fetch_trades(self, symbol, limit=50):
        kraken_symbol = self.SYMBOL_MAPPING.get(symbol, symbol)
        url = f"https://api.kraken.com/0/public/Trades?pair={kraken_symbol}"
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if 'error' in data and data['error']:
                logger.error(f"Kraken API error: {data['error']}")
                return pd.DataFrame()
            trades = data['result'].get(kraken_symbol, [])[:limit]
            df = pd.DataFrame(trades, columns=['price', 'volume', 'time', 'buy_sell', 'market_limit', 'misc'])
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
        except Exception as e:
            logger.error(f"Fetch trades failed: {e}")
            return pd.DataFrame()

    def get_order_book(self, symbol):
        kraken_symbol = self.SYMBOL_MAPPING.get(symbol, symbol)
        url = f"https://api.kraken.com/0/public/Depth?pair={kraken_symbol}&count=10"
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if 'error' in data and data['error']:
                logger.error(f"Kraken API error: {data['error']}")
                return {'bids': [], 'asks': [], 'best_bid': 0, 'best_ask': 0}
            result = data['result'].get(kraken_symbol, {})
            bids = [(float(price), float(volume)) for price, volume, _ in result.get('bids', [])]
            asks = [(float(price), float(volume)) for price, volume, _ in result.get('asks', [])]
            best_bid = bids[0][0] if bids else 0
            best_ask = asks[0][0] if asks else 0
            return {'bids': bids, 'asks': asks, 'best_bid': best_bid, 'best_ask': best_ask}
        except Exception as e:
            logger.error(f"Get order book failed: {e}")
            return {'bids': [], 'asks': [], 'best_bid': 0, 'best_ask': 0}

# Strategy implementations with proper DataFrame validation
class MeanReversionStrategy(TradingStrategy):
    def __init__(self, bollinger_window=20, std_dev=2.0, threshold=0.01):
        self.bollinger_window = bollinger_window
        self.std_dev = std_dev
        self.threshold = threshold
        self.performance = []

    def analyze(self, symbol, exchange_client, **kwargs):
        market_data = kwargs.get('market_data', {})
        bars_df = None
        
        if market_data and 'bars' in market_data:
            bars_df = market_data['bars']
            if not isinstance(bars_df, pd.DataFrame):
                return "HOLD", 0.0
        
        if bars_df is None or bars_df.empty:
            bars_df = exchange_client.fetch_bars(symbol, limit=self.bollinger_window + 10)
        
        if bars_df is None or not isinstance(bars_df, pd.DataFrame) or bars_df.empty or len(bars_df) < self.bollinger_window:
            return "HOLD", 0.0
        
        df = bars_df.copy()
        df['sma'] = df['close'].rolling(window=self.bollinger_window).mean()
        df['std'] = df['close'].rolling(window=self.bollinger_window).std()
        df['upper'] = df['sma'] + (df['std'] * self.std_dev)
        df['lower'] = df['sma'] - (df['std'] * self.std_dev)
        
        if pd.isna(df.iloc[-1]['sma']) or pd.isna(df.iloc[-1]['upper']) or pd.isna(df.iloc[-1]['lower']):
            return "HOLD", 0.0
        
        current_price = float(df['close'].iloc[-1])
        if current_price < df['lower'].iloc[-1] * (1 - self.threshold):
            return "BUY", current_price
        elif current_price > df['upper'].iloc[-1] * (1 + self.threshold):
            return "SELL", current_price
        return "HOLD", 0.0

    def get_metrics(self):
        if not self.performance:
            return {"win_rate": 0, "avg_return": 0, "sharpe": 0}
        return {
            "win_rate": sum(1 for p in self.performance if p > 0) / len(self.performance),
            "avg_return": np.mean(self.performance),
            "sharpe": (np.mean(self.performance) / (np.std(self.performance) + 1e-10) * np.sqrt(252))
        }

    def record_trade(self, trade_result):
        self.performance.append(trade_result)

class MomentumStrategy(TradingStrategy):
    def __init__(self, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9):
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.performance = []

    def analyze(self, symbol, exchange_client, **kwargs):
        market_data = kwargs.get('market_data', {})
        bars_df = None
        
        if market_data and 'bars' in market_data:
            bars_df = market_data['bars']
            if not isinstance(bars_df, pd.DataFrame):
                return "HOLD", 0.0
        
        if bars_df is None or bars_df.empty:
            bars_df = exchange_client.fetch_bars(symbol, limit=50)
        
        if bars_df is None or not isinstance(bars_df, pd.DataFrame) or bars_df.empty or len(bars_df) < max(self.rsi_period, self.macd_slow):
            return "HOLD", 0.0
        
        df = bars_df.copy()
        rsi_values = df.ta.rsi(close='close', length=self.rsi_period)
        rsi = rsi_values.iloc[-1] if not rsi_values.empty and not pd.isna(rsi_values.iloc[-1]) else 50.0
        
        macd = df.ta.macd(close='close', fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        macd_diff = 0.0
        if not macd.empty:
            macd_diff_col = f'MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}'
            if macd_diff_col in macd.columns:
                macd_diff = macd[macd_diff_col].iloc[-1] if not pd.isna(macd[macd_diff_col].iloc[-1]) else 0.0
        
        if rsi > 70 and macd_diff < 0:
            return "SELL", float(df['close'].iloc[-1])
        elif rsi < 30 and macd_diff > 0:
            return "BUY", float(df['close'].iloc[-1])
        return "HOLD", 0.0

    def get_metrics(self):
        if not self.performance:
            return {"win_rate": 0, "avg_return": 0, "sharpe": 0}
        return {
            "win_rate": sum(1 for p in self.performance if p > 0) / len(self.performance),
            "avg_return": np.mean(self.performance),
            "sharpe": (np.mean(self.performance) / (np.std(self.performance) + 1e-10) * np.sqrt(252))
        }

    def record_trade(self, trade_result):
        self.performance.append(trade_result)

class BreakoutStrategy(TradingStrategy):
    def __init__(self, atr_period=14, volume_spike_factor=1.5):
        self.atr_period = atr_period
        self.volume_spike_factor = volume_spike_factor
        self.performance = []

    def analyze(self, symbol, exchange_client, **kwargs):
        market_data = kwargs.get('market_data', {})
        bars_df = None
        
        if market_data and 'bars' in market_data:
            bars_df = market_data['bars']
            if not isinstance(bars_df, pd.DataFrame):
                return "HOLD", 0.0
        
        if bars_df is None or bars_df.empty:
            bars_df = exchange_client.fetch_bars(symbol, limit=50)
        
        if bars_df is None or not isinstance(bars_df, pd.DataFrame) or bars_df.empty or len(bars_df) < self.atr_period + 1:
            return "HOLD", 0.0
        
        df = bars_df.copy()
        atr = df.ta.atr(high='high', low='low', close='close', length=self.atr_period)
        atr = atr.iloc[-1] if not atr.empty and not pd.isna(atr.iloc[-1]) else 0.0
        
        current_price = float(df['close'].iloc[-1])
        prev_high = float(df['high'].iloc[-2])
        prev_low = float(df['low'].iloc[-2])
        prev_volume = float(df['volume'].iloc[-2])
        current_volume = float(df['volume'].iloc[-1])
        
        if current_price > prev_high + atr and current_volume > prev_volume * self.volume_spike_factor:
            return "BUY", current_price
        elif current_price < prev_low - atr and current_volume > prev_volume * self.volume_spike_factor:
            return "SELL", current_price
        return "HOLD", 0.0

    def get_metrics(self):
        if not self.performance:
            return {"win_rate": 0, "avg_return": 0, "sharpe": 0}
        return {
            "win_rate": sum(1 for p in self.performance if p > 0) / len(self.performance),
            "avg_return": np.mean(self.performance),
            "sharpe": (np.mean(self.performance) / (np.std(self.performance) + 1e-10) * np.sqrt(252))
        }

    def record_trade(self, trade_result):
        self.performance.append(trade_result)

class ArbitrageStrategy(TradingStrategy):
    def __init__(self, min_spread=0.01):
        self.min_spread = min_spread
        self.performance = []

    def analyze(self, symbol, exchange_client, **kwargs):
        symbol1, symbol2 = "BTC/USD", "ETH/USD"
        bars1 = exchange_client.fetch_bars(symbol1, limit=1)
        bars2 = exchange_client.fetch_bars(symbol2, limit=1)
        if bars1.empty or bars2.empty:
            return "HOLD", 0.0
        
        price1 = float(bars1['close'].iloc[-1])
        price2 = float(bars2['close'].iloc[-1])
        spread = abs(price1 - price2) / min(price1, price2)
        if spread > self.min_spread:
            return "BUY" if price1 < price2 else "SELL", price1
        return "HOLD", 0.0

    def get_metrics(self):
        if not self.performance:
            return {"win_rate": 0, "avg_return": 0, "sharpe": 0}
        return {
            "win_rate": sum(1 for p in self.performance if p > 0) / len(self.performance),
            "avg_return": np.mean(self.performance),
            "sharpe": (np.mean(self.performance) / (np.std(self.performance) + 1e-10) * np.sqrt(252))
        }

    def record_trade(self, trade_result):
        self.performance.append(trade_result)

class OptimizedMLStrategy(TradingStrategy):
    def __init__(self, lookback=60, num_estimators=100):
        self.lookback = lookback
        self.num_estimators = num_estimators
        self.model = None
        self.feature_importance = None
        self.performance = []
        self.feature_columns = ['close', 'high', 'low', 'open', 'volume']
        self.technical_indicators = ['rsi', 'macd', 'bollinger_b', 'atr', 'adx']

    def _preprocess_data(self, df):
        data = df.copy()
        data['rsi'] = data.ta.rsi(close='close', length=14).fillna(50)
        macd = data.ta.macd(close='close', fast=12, slow=26, signal=9)
        data['macd'] = macd['MACD_12_26_9'].fillna(0) if not macd.empty else 0
        bb = data.ta.bbands(close='close', length=20, std=2)
        data['bollinger_b'] = ((data['close'] - bb['BBL_20_2.0']) / (bb['BBU_20_2.0'] - bb['BBL_20_2.0'])).fillna(0.5) if not bb.empty else 0.5
        data['atr'] = data.ta.atr(high='high', low='low', close='close', length=14).fillna(0)
        data['adx'] = data.ta.adx(high='high', low='low', close='close', length=14)['ADX_14'].fillna(25)
        data['target'] = data['close'].pct_change(1).shift(-1).fillna(0)
        features = data[self.feature_columns + self.technical_indicators].values
        targets = data['target'].values
        return features, targets

    def train(self, symbol, exchange_client, force=False):
        bars_df = exchange_client.fetch_bars(symbol, TimeFrame.Minute, limit=self.lookback + 100)
        if bars_df.empty or len(bars_df) < self.lookback + 1:
            logger.error(f"Insufficient data for ML training on {symbol}")
            return False
        
        try:
            features, targets = self._preprocess_data(bars_df)
            X, y = [], []
            for i in range(len(features) - self.lookback - 1):
                X.append(features[i:i + self.lookback].flatten())
                y.append(targets[i + self.lookback])
            X, y = np.array(X), np.array(y)
            
            train_size = int(len(X) * 0.8)
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
            
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_valid = lgb.Dataset(X_val, y_val, reference=lgb_train)
            
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'learning_rate': 0.01,
                'max_depth': 6,
                'num_leaves': 63,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'verbosity': -1
            }
            
            self.model = lgb.train(params, lgb_train, num_boost_round=self.num_estimators, valid_sets=[lgb_valid])
            self.feature_importance = self.model.feature_importance()
            logger.info(f"LightGBM model trained for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error training LightGBM model: {e}")
            return False

    def analyze(self, symbol, exchange_client, **kwargs):
        market_data = kwargs.get('market_data', {})
        bars_df = None
        
        if market_data and 'bars' in market_data:
            bars_df = market_data['bars']
            if not isinstance(bars_df, pd.DataFrame):
                return "HOLD", 0.0
        
        if bars_df is None or bars_df.empty:
            bars_df = exchange_client.fetch_bars(symbol, limit=self.lookback + 10)
        
        if bars_df is None or not isinstance(bars_df, pd.DataFrame) or bars_df.empty or len(bars_df) < self.lookback:
            return "HOLD", 0.0
        
        if not self.model and not self.train(symbol, exchange_client):
            return "HOLD", 0.0
        
        try:
            features, _ = self._preprocess_data(bars_df)
            X = features[-self.lookback:].flatten().reshape(1, -1)
            prediction = self.model.predict(X)[0]
            current_price = float(bars_df['close'].iloc[-1])
            if prediction > 0.0025:
                return "BUY", current_price
            elif prediction < -0.0025:
                return "SELL", current_price
            return "HOLD", current_price
        except Exception as e:
            logger.error(f"Error in LightGBM prediction: {e}")
            return "HOLD", 0.0

    def get_metrics(self):
        if not self.performance:
            return {"win_rate": 0, "avg_return": 0, "sharpe": 0}
        return {
            "win_rate": sum(1 for p in self.performance if p > 0) / len(self.performance),
            "avg_return": np.mean(self.performance),
            "sharpe": (np.mean(self.performance) / (np.std(self.performance) + 1e-10) * np.sqrt(252))
        }

    def record_trade(self, trade_result):
        self.performance.append(trade_result)

# Enhanced WebSocket Trading
class EnhancedWebSocketTrading:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.ws = None
        self.order_status_callbacks = {}
        self.pending_orders = {}
        self.open_orders = {}
        self.order_updates = asyncio.Queue()
        self.heartbeat_task = None
        self.order_processor_task = None
        self.connect_lock = threading.Lock()
        self.req_id = 0

    async def connect(self):
        retry_count = 0
        while retry_count < 5:
            try:
                self.ws = websocket.WebSocketApp(
                    "wss://ws-auth.kraken.com",
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_open
                )
                self.ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
                self.ws_thread.start()
                timeout = time.time() + 10
                while not hasattr(self.ws, 'sock') or not self.ws.sock or not self.ws.sock.connected:
                    if time.time() > timeout:
                        raise ConnectionError("WebSocket connection timeout")
                    await asyncio.sleep(0.1)
                await self._authenticate()
                
                try:
                    current_loop = asyncio.get_running_loop()
                except RuntimeError:
                    current_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(current_loop)
                
                self.heartbeat_task = current_loop.create_task(self._heartbeat())
                self.order_processor_task = current_loop.create_task(self._process_order_updates())
                logger.info("WebSocket trading connected")
                return True
            except Exception as e:
                retry_count += 1
                logger.error(f"WebSocket connection error (attempt {retry_count}/5): {e}")
                await asyncio.sleep(2 ** retry_count)
        raise ConnectionError("Failed to connect after multiple attempts")

    async def _authenticate(self):
        nonce = str(int(time.time() * 1000))
        self.ws.send(json.dumps({"event": "subscribe", "subscription": {"name": "ownTrades"}}))

    async def _heartbeat(self):
        while True:
            try:
                if self.ws and self.ws.sock and self.ws.sock.connected:
                    self.ws.send(json.dumps({"op": "ping"}))
                await asyncio.sleep(15)
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(5)

    async def _process_order_updates(self):
        while True:
            try:
                update = await self.order_updates.get()
                order_id = update.get('order_id')
                if order_id in self.order_status_callbacks:
                    callback = self.order_status_callbacks[order_id]
                    await callback(update)
                self.order_updates.task_done()
            except Exception as e:
                logger.error(f"Error processing order update: {e}")
                await asyncio.sleep(0.1)

    def _on_open(self, ws):
        logger.info("WebSocket trading connection opened")

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            if isinstance(data, list) and data[0] == "ownTrades":
                for trade_id, trade_info in data[1].items():
                    order_id = trade_info.get("orderTxid")
                    if order_id in self.pending_orders:
                        asyncio.run_coroutine_threadsafe(
                            self.order_updates.put({'order_id': order_id, 'trades': [trade_info], 'success': True}),
                            asyncio.get_event_loop()
                        )
        except Exception as e:
            logger.error(f"WebSocket message processing error: {e}")

    def _on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")

    async def place_order_with_timeout(self, symbol, order_type, side, size, price=None, timeout=5.0):
        order_placed = asyncio.Event()
        order_result = {'success': False, 'reason': 'Timeout', 'order_id': None, 'trades': []}
        
        async def order_callback(result):
            nonlocal order_result
            order_result = result
            order_placed.set()
        
        order_id = self.place_order(symbol, order_type, side, size, price, callback=order_callback)
        try:
            await asyncio.wait_for(order_placed.wait(), timeout)
            return order_result
        except asyncio.TimeoutError:
            logger.warning(f"Order placement timeout for {symbol} {side}")
            return {'success': False, 'reason': 'Timeout', 'order_id': order_id, 'trades': []}

    def place_order(self, symbol, order_type, side, volume, price=None, callback=None):
        self.req_id += 1
        req_id = str(self.req_id)
        params = {"ordertype": order_type, "type": side.lower(), "pair": symbol, "volume": str(volume)}
        if price and order_type != "market":
            params["price"] = str(price)
        self.ws.send(json.dumps({"name": "addOrder", "reqid": req_id, "params": params}))
        order_id = f"pending_{req_id}"
        self.pending_orders[order_id] = {"symbol": symbol, "type": order_type, "side": side, "volume": volume, "price": price}
        if callback:
            self.order_status_callbacks[order_id] = callback
        return order_id

# Simplified supporting classes
class Level1Features:
    def __init__(self, window_sizes=[5, 10, 30, 60]):
        self.window_sizes = window_sizes
        self.bid_ask_history = defaultdict(list)
        self.max_history = max(window_sizes) * 2

    def update(self, symbol, timestamp, bid, ask, bid_size, ask_size):
        data_point = {
            'timestamp': timestamp, 'bid': bid, 'ask': ask, 'mid': (bid + ask) / 2,
            'spread': ask - bid, 'spread_pct': (ask - bid) / bid if bid > 0 else 0,
            'bid_size': bid_size, 'ask_size': ask_size,
            'imbalance': (bid_size - ask_size) / (bid_size + ask_size) if (bid_size + ask_size) > 0 else 0
        }
        self.bid_ask_history[symbol].append(data_point)
        if len(self.bid_ask_history[symbol]) > self.max_history:
            self.bid_ask_history[symbol].pop(0)

    def get_features(self, symbol):
        if not self.bid_ask_history[symbol]:
            return {}
        history = self.bid_ask_history[symbol]
        features = {
            'spread': history[-1]['spread'], 'spread_pct': history[-1]['spread_pct'], 'imbalance': history[-1]['imbalance']
        }
        return features

class OrderExecutionAlgorithms:
    def __init__(self, exchange_client, trading_client):
        self.exchange_client = exchange_client
        self.trading_client = trading_client
        self.active_algorithms = {}
        self.algo_id_counter = 0

    def twap(self, symbol, side, total_quantity, duration_seconds, max_participation=0.3, callback=None):
        self.algo_id_counter += 1
        algo_id = f"twap_{self.algo_id_counter}"
        asyncio.run(self.trading_client.place_order_with_timeout(symbol, "market", side, total_quantity))
        if callback:
            callback({"status": "completed", "executed_quantity": total_quantity, "trades": [{"vol": total_quantity, "price": 0}]})
        return algo_id

    def start_execution_thread(self):
        pass

    def stop_execution_thread(self):
        pass

class LiquidityManager:
    def __init__(self, exchange_client):
        self.exchange_client = exchange_client

    def check_liquidity(self, symbol, order_size, side, max_impact=0.01, min_depth_ratio=10):
        return {"is_liquid": True, "price_impact": 0.01, "depth_ratio": 20, "weighted_price": 0}

class AsyncAPIManager:
    def __init__(self, exchange_client, max_concurrent=3):
        self.exchange_client = exchange_client
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.cache = {}

    def request(self, method, *args, **kwargs):
        return getattr(self.exchange_client, method)(*args, **kwargs)

    def fetch_multiple_bars(self, symbols, timeframe=TimeFrame.Minute, limit=50, callback=None):
        result = {symbol: self.request("fetch_bars", symbol, timeframe, limit) for symbol in symbols}
        if callback:
            callback(result)
        return result

    def start(self):
        pass

    def stop(self):
        pass

class StrategyPerformanceTracker:
    def __init__(self, strategy_name):
        self.strategy_name = strategy_name
        self.trades = []

    def add_trade(self, trade):
        self.trades.append(trade)

    def get_metrics(self):
        if not self.trades:
            return {'total_trades': 0, 'win_rate': 0, 'profit_factor': 0}
        profits = [t['pnl'] for t in self.trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in self.trades if t['pnl'] <= 0]
        total_trades = len(self.trades)
        win_rate = len(profits) / total_trades if total_trades > 0 else 0
        profit_factor = sum(profits) / abs(sum(losses)) if losses else float('inf')
        return {'total_trades': total_trades, 'win_rate': win_rate, 'profit_factor': profit_factor}

    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            json.dump({'strategy': self.strategy_name, 'trades': self.trades}, f)

# Main bot class
class KrakenCryptoLogger:
    def __init__(self, config_path="config.json", exchange_client=None, state_file="bot_state.json"):
        self.logger = logger
        self._load_config(config_path)
        self._initialize_google_sheets()
        
        self.exchange_client = exchange_client or KrakenClient(KRAKEN_API_KEY, KRAKEN_PRIVATE_KEY)
        self.ws_trading = EnhancedWebSocketTrading(KRAKEN_API_KEY, KRAKEN_PRIVATE_KEY)
        self.execution_algorithms = OrderExecutionAlgorithms(self.exchange_client, self.ws_trading)
        self.liquidity_manager = LiquidityManager(self.exchange_client)
        self.api_manager = AsyncAPIManager(self.exchange_client)
        self.level1_analyzer = Level1Features()
        
        self.active_symbols = self.config.get("symbols", ["BTC/USD", "ETH/USD"])
        self.state_file = state_file
        self.lookback = 60
        
        self.realtime_data_lock = threading.Lock()
        self.shutdown_event = threading.Event()
        
        self._initialize_fresh_state()  # Initialize state before strategies
        
        # Initialize strategies after state
        self.strategies = {
            "mean_reversion": MeanReversionStrategy(
                self.config["strategies"]["mean_reversion"]["bollinger_window"],
                self.config["strategies"]["mean_reversion"]["std_dev"],
                self.config["strategies"]["mean_reversion"]["threshold"]
            ),
            "momentum": MomentumStrategy(
                self.config["strategies"]["momentum"]["rsi_period"],
                self.config["strategies"]["momentum"]["macd_fast"],
                self.config["strategies"]["momentum"]["macd_slow"],
                self.config["strategies"]["momentum"]["macd_signal"]
            ),
            "breakout": BreakoutStrategy(
                self.config["strategies"]["breakout"]["atr_period"],
                self.config["strategies"]["breakout"]["volume_spike_factor"]
            ),
            "arbitrage": ArbitrageStrategy(
                self.config["strategies"]["arbitrage"]["min_spread"]
            ),
            "ml_prediction": OptimizedMLStrategy(lookback=60, num_estimators=100)
        }
        
        self.performance_trackers = {name: StrategyPerformanceTracker(name) for name in self.strategies.keys()}
        
        if os.path.exists(state_file) and self.load_state():
            logger.info("Loaded previous state")
        
        self._initialize_price_data()
        
        self.running = False
        self.api_manager.start()
        asyncio.run(self.ws_trading.connect())

    def _load_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}. Using default config.")
            self.config = {
                "symbols": ["BTC/USD", "ETH/USD"],
                "strategies": {
                    "mean_reversion": {"bollinger_window": 20, "std_dev": 2.0, "threshold": 0.01},
                    "momentum": {"rsi_period": 14, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9},
                    "breakout": {"atr_period": 14, "volume_spike_factor": 1.5},
                    "arbitrage": {"min_spread": 0.01}
                },
                "leverage": {"max_leverage": 3.0, "initial_leverage": 1.0},
                "websocket": {"subscription": ["ticker"]}
            }

    def _initialize_google_sheets(self):
        if not os.path.exists(GOOGLE_CREDENTIALS_FILE):
            self.sheets_enabled = False
            return
        try:
            scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
            creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CREDENTIALS_FILE, scope)
            self.gc = gspread.authorize(creds)
            self.spreadsheet = self.gc.open("ALPACA CRYPTO BACKTESTER")
            self.trade_log_sheet = self.spreadsheet.worksheet("Trade Log")
            self.sheets_enabled = True
        except Exception as e:
            logger.warning(f"Google Sheets initialization failed: {e}")
            self.sheets_enabled = False

    def _initialize_fresh_state(self):
        self.active_trades = {}
        self.trade_id_counter = 1000
        self.realtime_prices = defaultdict(list)
        self.price_data = defaultdict(list)
        self.order_flow = defaultdict(list)
        strategy_names = ['mean_reversion', 'momentum', 'breakout', 'arbitrage', 'ml_prediction']
        self.strategy_weights = {name: 0.2 for name in strategy_names}
        self.portfolio = {"cash": 10000.0, "positions": {}, "returns": [], "trade_performance": defaultdict(list)}
        self.risk_per_trade = 0.01
        self.var_limit = 0.05
        self.volatility_threshold = 50

    def _initialize_price_data(self):
        for symbol in self.active_symbols:
            historical_bars = self.api_manager.request("fetch_bars", symbol, TimeFrame.Minute, 30)
            if not historical_bars.empty:
                for _, row in historical_bars.iterrows():
                    self.price_data[symbol].append({"timestamp": row['timestamp'], "close": float(row['close']), "volume": float(row['volume'])})

    def dynamic_position_sizing(self, symbol, entry_price):
        return self.portfolio["cash"] * self.risk_per_trade / entry_price if entry_price > 0 else 0

    def manage_leverage(self, symbol, position_size):
        return min(self.config["leverage"]["max_leverage"], 1.0)

    def execute_trade(self, symbol, action, price, leverage=1.0, size=None, max_hold_time=3600):
        try:
            trade_id = f"T{self.trade_id_counter + 1}"
            self.trade_id_counter += 1
            
            if size is None:
                size = self.dynamic_position_sizing(symbol, price)
            
            if size <= 0 or self.portfolio["cash"] < size * price:
                self.logger.warning(f"Insufficient funds for {symbol} trade")
                return None
            
            liquidity_check = self.liquidity_manager.check_liquidity(symbol, size, action.lower())
            if not liquidity_check["is_liquid"]:
                size = size * 0.5
                if size <= 0 or self.portfolio["cash"] < size * price:
                    return None
            
            trade = {
                "symbol": symbol, "action": action, "entry_price": price, "leverage": leverage,
                "size": size, "entry_time": time.time(), "max_hold_time": max_hold_time,
                "status": "PENDING", "strategy": "unknown", "execution_strategy": "market", "fills": []
            }
            self.active_trades[trade_id] = trade
            
            result = asyncio.run(self.ws_trading.place_order_with_timeout(symbol, "market", action.lower(), size, timeout=5.0))
            self._handle_trade_execution(trade_id, result)
            self.logger.info(f"Initiated {action} trade {trade_id}")
            return trade_id
        except Exception as e:
            self.logger.error(f"Error executing trade for {symbol}: {e}")
            return None

    def _handle_trade_execution(self, trade_id, order_result):
        if trade_id not in self.active_trades:
            return
        trade = self.active_trades[trade_id]
        if order_result.get("success"):
            executed_qty = sum(float(t.get("vol", 0)) for t in order_result.get("trades", []))
            avg_price = sum(float(t.get("price", 0)) * float(t.get("vol", 0)) 
                           for t in order_result.get("trades", [])) / executed_qty if executed_qty > 0 else trade["entry_price"]
            trade["entry_price"] = avg_price
            trade["size"] = executed_qty
            trade["status"] = "OPEN"
            trade["fills"].extend(order_result.get("trades", []))
            self.update_portfolio(trade_id)
            self.logger.info(f"Trade {trade_id} executed: {executed_qty} {trade['symbol']} at {avg_price}")
        else:
            trade["status"] = "CANCELLED"
            self.logger.warning(f"Trade {trade_id} cancelled: {order_result.get('reason', 'Unknown')}")
            del self.active_trades[trade_id]

    def monitor_trades(self):
        current_time = time.time()
        for trade_id, trade in list(self.active_trades.items()):
            if trade["status"] not in ["OPEN", "PENDING"]:
                continue
            
            symbol = trade["symbol"]
            bars = self.api_manager.request("fetch_bars", symbol, TimeFrame.Minute, 1)
            if bars.empty:
                continue
            current_price = float(bars['close'].iloc[-1])
            
            if trade["status"] == "PENDING" and current_time - trade["entry_time"] > 300:
                trade["status"] = "EXPIRED"
                self.update_portfolio(trade_id)
                continue
            
            if trade["action"] == "BUY" and current_price <= trade["entry_price"] * 0.97:
                trade["status"] = "STOPPED"
                trade["exit_price"] = current_price
                self.update_portfolio(trade_id)
            elif trade["action"] == "SELL" and current_price >= trade["entry_price"] * 1.03:
                trade["status"] = "STOPPED"
                trade["exit_price"] = current_price
                self.update_portfolio(trade_id)
            elif current_time - trade["entry_time"] >= trade["max_hold_time"]:
                trade["status"] = "EXPIRED"
                trade["exit_price"] = current_price
                self.update_portfolio(trade_id)
            elif trade["action"] == "BUY" and current_price >= trade["entry_price"] * 1.05:
                trade["status"] = "PROFIT_TAKEN"
                trade["exit_price"] = current_price
                self.update_portfolio(trade_id)
            elif trade["action"] == "SELL" and current_price <= trade["entry_price"] * 0.95:
                trade["status"] = "PROFIT_TAKEN"
                trade["exit_price"] = current_price
                self.update_portfolio(trade_id)

    def evaluate_strategies(self):
        strategy_metrics = {}
        for name, tracker in self.performance_trackers.items():
            metrics = tracker.get_metrics()
            if metrics['total_trades'] < 5:
                continue
            performance_score = metrics['win_rate'] * 0.5 + metrics['profit_factor'] * 0.5
            strategy_metrics[name] = {"performance_score": performance_score}
        
        if strategy_metrics:
            total_score = sum(metrics["performance_score"] for _, metrics in strategy_metrics.items())
            for strategy, metrics in strategy_metrics.items():
                self.strategy_weights[strategy] = max(0.1, min(0.6, metrics["performance_score"] / total_score))
            self.logger.info(f"Updated strategy weights: {self.strategy_weights}")

    def check_risk_limits(self):
        var = self.calculate_var()
        if var > self.var_limit * self.portfolio["cash"]:
            self.logger.warning(f"VaR limit exceeded: {var:.2f}")
            return False
        return True

    def calculate_var(self):
        if not self.portfolio["returns"]:
            return 0.0
        returns = np.array(self.portfolio["returns"])
        return abs(norm.ppf(0.05, np.mean(returns), np.std(returns)) * self.portfolio["cash"])

    def start_websocket(self):
        if hasattr(self, 'websocket') and self.websocket:
            try:
                self.websocket.close()
            except:
                pass
        
        if hasattr(self, 'websocket_thread') and self.websocket_thread and self.websocket_thread.is_alive():
            if threading.current_thread() != self.websocket_thread:
                try:
                    self.websocket_thread.join(timeout=1)
                except:
                    pass
        
        ws_url = "wss://ws.kraken.com"
        self.websocket = websocket.WebSocketApp(
            ws_url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open
        )
        self.websocket_thread = threading.Thread(target=self.websocket.run_forever, daemon=True)
        self.websocket_thread.start()
        self.logger.info("WebSocket thread started")

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            if "event" in data or not isinstance(data, list):
                return
            symbol = data[-1]
            price = float(data[1]["c"][0])
            bid = float(data[1]["b"][0])
            ask = float(data[1]["a"][0])
            bid_size = float(data[1]["b"][1])
            ask_size = float(data[1]["a"][1])
            timestamp = time.time()
            self.level1_analyzer.update(symbol, timestamp, bid, ask, bid_size, ask_size)
            bar_data = {"timestamp": timestamp, "close": price, "volume": 0.0}
            self.price_data[symbol].append(bar_data)
            self.log_realtime_data(symbol, bar_data)
        except Exception as e:
            self.logger.error(f"Error processing WebSocket message: {e}")

    def _on_error(self, ws, error):
        self.logger.error(f"WebSocket error: {error}. Restarting...")
        self.stop_websocket()
        time.sleep(5)
        self.start_websocket()

    def _on_close(self, ws, close_status_code, close_msg):
        self.logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")

    def _on_open(self, ws):
        subscription = {"event": "subscribe", "pair": self.active_symbols, "subscription": {"name": "ticker"}}
        ws.send(json.dumps(subscription))
        self.logger.info("WebSocket subscribed to ticker")

    def stop_websocket(self):
        if hasattr(self, 'websocket') and self.websocket:
            try:
                self.websocket.close()
            except:
                pass
        if hasattr(self, 'websocket_thread') and self.websocket_thread and self.websocket_thread.is_alive():
            if threading.current_thread() != self.websocket_thread:
                try:
                    self.websocket_thread.join(timeout=1)
                except:
                    pass
        self.websocket = None
        self.websocket_thread = None
        self.logger.info("WebSocket stopped")

    def log_realtime_data(self, symbol, bar_data):
        with self.realtime_data_lock:
            self.realtime_prices[symbol].append(bar_data)
            if len(self.realtime_prices[symbol]) > 10:
                self.realtime_prices[symbol].pop(0)

    def save_state(self):
        try:
            state = {
                "trade_id_counter": self.trade_id_counter,
                "strategy_weights": self.strategy_weights,
                "portfolio": {
                    "cash": float(self.portfolio["cash"]),
                    "positions": self.portfolio["positions"],
                    "returns": self.portfolio["returns"],
                    "trade_performance": dict(self.portfolio["trade_performance"])
                },
                "active_trades": self.active_trades
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            self.logger.info(f"Bot state saved to {self.state_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
            return False

    def load_state(self):
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            self.trade_id_counter = state.get("trade_id_counter", 1000)
            self.strategy_weights = state.get("strategy_weights", self.strategy_weights)
            self.portfolio = state.get("portfolio", {"cash": 10000.0, "positions": {}, "returns": [], "trade_performance": defaultdict(list)})
            self.active_trades = state.get("active_trades", {})
            self.realtime_prices = defaultdict(list)
            self.price_data = defaultdict(list)
            self.order_flow = defaultdict(list)
            return True
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")
            return False

    def run_trading_loop(self):
        if self.running:
            self.logger.warning("Trading loop already running")
            return
        self.running = True
        websocket_reconnect_time = 0
        max_retries = 3
        retry_count = 0
        
        try:
            for symbol in self.active_symbols:
                self.strategies["ml_prediction"].train(symbol, self.exchange_client)
            
            self.start_websocket()
            self.execution_algorithms.start_execution_thread()
            self.strategy_eval_counter = 0
            self.last_eval_time = 0
            
            while not self.shutdown_event.is_set():
                current_time = time.time()
                
                if (not hasattr(self, 'websocket') or not self.websocket) and current_time - websocket_reconnect_time > 30:
                    self.logger.info("WebSocket connection lost, attempting to restart...")
                    self.stop_websocket()
                    time.sleep(5)
                    self.start_websocket()
                    websocket_reconnect_time = current_time
                
                if not self.check_risk_limits():
                    self.logger.info("Risk limits triggered. Pausing for 300s")
                    time.sleep(300)
                    continue
                
                self.monitor_trades()
                
                try:
                    self.api_manager.fetch_multiple_bars(
                        self.active_symbols, TimeFrame.Minute, limit=60,
                        callback=self._process_trading_signals_safe
                    )
                    retry_count = 0
                except Exception as e:
                    retry_count += 1
                    self.logger.error(f"Error fetching market data: {e}, retry {retry_count}/{max_retries}")
                    if retry_count >= max_retries:
                        self.save_state()
                        time.sleep(30)
                        retry_count = 0
                
                self.logger.info(f"Portfolio Cash: {self.portfolio['cash']:.2f}, Positions: {len(self.portfolio['positions'])}")
                time.sleep(5)
                
        except KeyboardInterrupt:
            self.logger.info("Trading loop interrupted")
        except Exception as e:
            self.logger.error(f"Unexpected error in trading loop: {e}", exc_info=True)
        finally:
            self.save_state()
            self.stop_websocket()
            self.execution_algorithms.stop_execution_thread()
            self.api_manager.stop()
            self.running = False

    def _process_trading_signals_safe(self, market_data):
        try:
            self._process_trading_signals(market_data)
        except Exception as e:
            self.logger.error(f"Error processing trading signals: {e}", exc_info=True)

    def _process_trading_signals(self, market_data):
        if not market_data:
            return
        
        self.strategy_eval_counter += 1
        current_time = time.time()
        if self.strategy_eval_counter >= 10 or (current_time - self.last_eval_time) > 86400:
            self.evaluate_strategies()
            self.strategy_eval_counter = 0
            self.last_eval_time = current_time
        
        for symbol in self.active_symbols:
            if symbol in self.portfolio["positions"]:
                continue
                
            if symbol not in market_data:
                continue
                
            symbol_data = market_data.get(symbol)
            if symbol_data is None or (hasattr(symbol_data, 'empty') and symbol_data.empty):
                continue
            
            market_data_package = {'bars': symbol_data}
            
            try:
                signals = {}
                for name, strategy in self.strategies.items():
                    signals[name] = strategy.analyze(symbol, self.exchange_client, market_data=market_data_package)
                
                weighted_votes = {"BUY": 0, "SELL": 0, "HOLD": 0}
                for name, (action, _) in signals.items():
                    weighted_votes[action] += self.strategy_weights.get(name, 0.2)
                
                action = max(weighted_votes.items(), key=lambda x: x[1])[0]
                if action in ["BUY", "SELL"]:
                    contributing_strategies = [(name, self.strategy_weights.get(name, 0) if signals[name][0] == action else 0)
                                              for name in self.strategies.keys()]
                    contributing_strategy = max(contributing_strategies, key=lambda x: x[1])[0]
                    
                    _, price = signals[contributing_strategy]
                    if price <= 0:
                        if not symbol_data.empty and len(symbol_data) > 0:
                            price = float(symbol_data['close'].iloc[-1])
                        else:
                            continue
                    
                    trade_id = self.execute_trade(symbol, action, price)
                    if trade_id:
                        self.active_trades[trade_id]["strategy"] = contributing_strategy
            except Exception as e:
                self.logger.error(f"Error processing signals for {symbol}: {e}", exc_info=True)
        
        if int(current_time) % 900 < 10:
            self.save_state()

    def update_portfolio(self, trade_id):
        trade = self.active_trades.get(trade_id)
        if not trade:
            return
        symbol = trade["symbol"]
        action = trade["action"]
        entry_price = trade["entry_price"]
        size = trade["size"]
        strategy = trade.get("strategy", "unknown")
        
        if trade["status"] == "OPEN" and action == "BUY":
            cost = size * entry_price
            if cost <= self.portfolio["cash"]:
                self.portfolio["cash"] -= cost
                self.portfolio["positions"][symbol] = {"size": size, "entry_price": entry_price, "trade_id": trade_id, "strategy": strategy}
        
        elif trade["status"] in ["STOPPED", "EXPIRED", "PROFIT_TAKEN"]:
            exit_price = trade["exit_price"]
            if symbol in self.portfolio["positions"]:
                position = self.portfolio["positions"][symbol]
                profit = (exit_price - entry_price) * size if action == "BUY" else (entry_price - exit_price) * size
                self.portfolio["cash"] += (size * exit_price) + profit
                self.portfolio["returns"].append(profit / (size * entry_price))
                self.performance_trackers[strategy].add_trade({
                    "trade_id": trade_id, "symbol": symbol, "entry_time": trade["entry_time"],
                    "exit_time": time.time(), "entry_price": entry_price, "exit_price": exit_price,
                    "size": size, "profit": profit, "status": trade["status"], "pnl": profit
                })
                self.strategies[strategy].record_trade(profit)
                del self.portfolio["positions"][symbol]
            del self.active_trades[trade_id]

if __name__ == "__main__":
    max_restarts = 3
    restart_count = 0
    
    while restart_count < max_restarts:
        try:
            if not KRAKEN_API_KEY or not KRAKEN_PRIVATE_KEY or "YOUR_API_KEY_HERE" in KRAKEN_API_KEY:
                logger.error("Invalid Kraken API credentials. Please set KRAKEN_API_KEY and KRAKEN_PRIVATE_KEY environment variables or update the script.")
                sys.exit(1)

            config_path = "config.json"
            if not os.path.exists(config_path):
                default_config = {
                    "symbols": ["BTC/USD", "ETH/USD"],
                    "strategies": {
                        "mean_reversion": {"bollinger_window": 20, "std_dev": 2.0, "threshold": 0.01},
                        "momentum": {"rsi_period": 14, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9},
                        "breakout": {"atr_period": 14, "volume_spike_factor": 1.5},
                        "arbitrage": {"min_spread": 0.01}
                    },
                    "leverage": {"max_leverage": 3.0, "initial_leverage": 1.0},
                    "websocket": {"subscription": ["ticker"]}
                }
                with open(config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                logger.info(f"Created default config at {config_path}")

            bot = KrakenCryptoLogger(
                config_path=config_path,
                exchange_client=KrakenClient(KRAKEN_API_KEY, KRAKEN_PRIVATE_KEY),
                state_file="bot_state.json"
            )
            
            logger.info("Starting trading bot")
            bot.run_trading_loop()
            break
            
        except Exception as e:
            restart_count += 1
            logger.error(f"Main execution failed (attempt {restart_count}/{max_restarts}): {e}", exc_info=True)
            if 'bot' in locals():
                bot.save_state()
                bot.stop_websocket()
                bot.execution_algorithms.stop_execution_thread()
                bot.api_manager.stop()
            
            if restart_count < max_restarts:
                logger.info(f"Restarting in 30 seconds...")
                time.sleep(30)
            else:
                logger.error("Maximum restart attempts reached. Exiting.")
                sys.exit(1)