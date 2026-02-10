import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCollector:
    """
    Recolector de datos de exchanges de crypto
    Soporta múltiples exchanges vía CCXT
    """
    def __init__(self, exchange_name='binance', api_key=None, api_secret=None):
        self.exchange_name = exchange_name
        
        # Inicializar exchange
        exchange_class = getattr(ccxt, exchange_name)
        
        if api_key and api_secret:
            self.exchange = exchange_class({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
            })
        else:
            self.exchange = exchange_class({
                'enableRateLimit': True,
            })
            
        logger.info(f"Conectado a {exchange_name}")
        
    def get_historical_data(self, symbol='SOL/USDT', timeframe='5m', limit=1000):
        """
        Obtener datos históricos de velas
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            logger.info(f"Descargados {len(df)} velas de {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error obteniendo datos históricos: {e}")
            return None
    
    def get_current_price(self, symbol='SOL/USDT'):
        """Obtener precio actual"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker['last'])
        except Exception as e:
            logger.error(f"Error obteniendo precio: {e}")
            return None
    
    def get_orderbook(self, symbol='SOL/USDT', limit=20):
        """Obtener libro de órdenes"""
        try:
            orderbook = self.exchange.fetch_order_book(symbol, limit)
            return orderbook
        except Exception as e:
            logger.error(f"Error obteniendo orderbook: {e}")
            return None
    
    def get_balance(self):
        """Obtener balance de la cuenta"""
        try:
            balance = self.exchange.fetch_balance()
            return balance
        except Exception as e:
            logger.error(f"Error obteniendo balance: {e}")
            return None
    
    def place_order(self, symbol, order_type, side, amount, price=None):
        """
        Colocar orden
        order_type: 'market' o 'limit'
        side: 'buy' o 'sell'
        """
        try:
            if order_type == 'market':
                order = self.exchange.create_market_order(symbol, side, amount)
            else:
                order = self.exchange.create_limit_order(symbol, side, amount, price)
                
            logger.info(f"Orden colocada: {side} {amount} {symbol} @ {price if price else 'market'}")
            return order
            
        except Exception as e:
            logger.error(f"Error colocando orden: {e}")
            return None
    
    def get_open_orders(self, symbol='SOL/USDT'):
        """Obtener órdenes abiertas"""
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            return orders
        except Exception as e:
            logger.error(f"Error obteniendo órdenes abiertas: {e}")
            return []
    
    def cancel_order(self, order_id, symbol='SOL/USDT'):
        """Cancelar orden"""
        try:
            result = self.exchange.cancel_order(order_id, symbol)
            logger.info(f"Orden {order_id} cancelada")
            return result
        except Exception as e:
            logger.error(f"Error cancelando orden: {e}")
            return None


class MarketAnalyzer:
    """
    Analizador de mercado en tiempo real
    Detecta patrones y señales de trading
    """
    def __init__(self):
        self.price_history = []
        self.volume_history = []
        
    def update(self, price, volume):
        """Actualizar con nuevo dato"""
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        # Mantener solo últimos 1000 datos
        if len(self.price_history) > 1000:
            self.price_history.pop(0)
            self.volume_history.pop(0)
    
    def detect_trend(self, window=20):
        """
        Detectar tendencia del mercado
        Returns: 'uptrend', 'downtrend', 'sideways'
        """
        if len(self.price_history) < window:
            return 'sideways'
        
        recent_prices = self.price_history[-window:]
        
        # Calcular pendiente
        x = np.arange(len(recent_prices))
        slope = np.polyfit(x, recent_prices, 1)[0]
        
        # Calcular volatilidad
        volatility = np.std(recent_prices) / np.mean(recent_prices)
        
        if abs(slope) < volatility * 0.1:
            return 'sideways'
        elif slope > 0:
            return 'uptrend'
        else:
            return 'downtrend'
    
    def calculate_support_resistance(self, window=100):
        """
        Calcular niveles de soporte y resistencia
        """
        if len(self.price_history) < window:
            return None, None
        
        recent_prices = self.price_history[-window:]
        
        # Encontrar picos y valles locales
        peaks = []
        valleys = []
        
        for i in range(1, len(recent_prices) - 1):
            if recent_prices[i] > recent_prices[i-1] and recent_prices[i] > recent_prices[i+1]:
                peaks.append(recent_prices[i])
            elif recent_prices[i] < recent_prices[i-1] and recent_prices[i] < recent_prices[i+1]:
                valleys.append(recent_prices[i])
        
        resistance = np.mean(sorted(peaks)[-3:]) if len(peaks) >= 3 else max(recent_prices)
        support = np.mean(sorted(valleys)[:3]) if len(valleys) >= 3 else min(recent_prices)
        
        return support, resistance
    
    def detect_volume_spike(self, threshold=2.0):
        """
        Detectar picos de volumen
        """
        if len(self.volume_history) < 20:
            return False
        
        avg_volume = np.mean(self.volume_history[-20:-1])
        current_volume = self.volume_history[-1]
        
        return current_volume > avg_volume * threshold
    
    def calculate_volatility(self, window=20):
        """Calcular volatilidad reciente"""
        if len(self.price_history) < window:
            return 0
        
        recent_prices = self.price_history[-window:]
        returns = np.diff(recent_prices) / recent_prices[:-1]
        
        return np.std(returns)


class SignalGenerator:
    """
    Generador de señales de trading basado en múltiples estrategias
    """
    def __init__(self):
        self.signals = []
        
    def generate_signal(self, features, ml_prediction, market_context):
        """
        Generar señal de trading combinando ML y análisis técnico
        
        Args:
            features: características técnicas actuales
            ml_prediction: [buy_prob, hold_prob, sell_prob]
            market_context: contexto del mercado (trend, support, resistance, etc.)
        
        Returns:
            signal: 'BUY', 'SELL', 'HOLD'
            confidence: 0-1
        """
        buy_prob, hold_prob, sell_prob = ml_prediction
        
        # Factor de tendencia
        trend_factor = 1.0
        if market_context.get('trend') == 'uptrend':
            trend_factor = 1.2
        elif market_context.get('trend') == 'downtrend':
            trend_factor = 0.8
        
        # Ajustar probabilidades con contexto de mercado
        adjusted_buy = buy_prob * trend_factor
        adjusted_sell = sell_prob / trend_factor
        
        # Consideraciones de soporte/resistencia
        current_price = market_context.get('current_price', 0)
        support = market_context.get('support', 0)
        resistance = market_context.get('resistance', float('inf'))
        
        # Boost buy si estamos cerca del soporte
        if support > 0 and current_price < support * 1.02:
            adjusted_buy *= 1.3
        
        # Boost sell si estamos cerca de la resistencia
        if resistance < float('inf') and current_price > resistance * 0.98:
            adjusted_sell *= 1.3
        
        # Determinar señal
        max_prob = max(adjusted_buy, hold_prob, adjusted_sell)
        
        if max_prob == adjusted_buy and adjusted_buy > 0.5:
            signal = 'BUY'
            confidence = adjusted_buy
        elif max_prob == adjusted_sell and adjusted_sell > 0.5:
            signal = 'SELL'
            confidence = adjusted_sell
        else:
            signal = 'HOLD'
            confidence = hold_prob
        
        # Logging
        logger.info(f"Signal: {signal} | Confidence: {confidence:.2f} | Trend: {market_context.get('trend')}")
        
        self.signals.append({
            'timestamp': datetime.now(),
            'signal': signal,
            'confidence': confidence,
            'ml_prediction': ml_prediction,
            'market_context': market_context
        })
        
        return signal, confidence
