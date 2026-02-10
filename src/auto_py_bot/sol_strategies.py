"""
Estrategias específicas que funcionan bien con SOL/USDT
Basadas en características únicas del comportamiento de Solana
"""

import numpy as np
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class SOLStrategies:
    """
    Colección de estrategias específicas para SOL
    """
    
    @staticmethod
    def momentum_breakout(features: np.ndarray, market_context: dict) -> Tuple[str, float]:
        """
        Estrategia de Momentum Breakout
        
        SOL tiende a tener movimientos explosivos cuando:
        - RSI < 30 (oversold) con volumen creciente
        - Precio rompe resistencia con volumen alto
        - MACD cruza hacia arriba
        
        Returns: (signal, confidence)
        """
        # Extraer indicadores del último punto
        current_price = market_context.get('current_price', 0)
        resistance = market_context.get('resistance', float('inf'))
        support = market_context.get('support', 0)
        volatility = market_context.get('volatility', 0)
        
        # Asumiendo que features tiene: [price, rsi, macd, signal, sma_dist, bb_pos, vol_ratio, momentum, volatility, range]
        rsi = features[1] if len(features) > 1 else 50
        macd = features[2] if len(features) > 2 else 0
        signal_line = features[3] if len(features) > 3 else 0
        volume_ratio = features[6] if len(features) > 6 else 1
        momentum = features[7] if len(features) > 7 else 0
        
        signal = 'HOLD'
        confidence = 0.5
        
        # COMPRA: Breakout alcista
        if (rsi < 35 and  # Oversold pero no extremo
            volume_ratio > 1.5 and  # Volumen alto
            macd > signal_line and  # MACD cruzando arriba
            momentum > 0 and  # Momentum positivo
            current_price > support * 1.01):  # Por encima del soporte
            
            signal = 'BUY'
            confidence = min(0.85, 0.6 + (volume_ratio - 1.5) * 0.1)
            logger.info(f"🚀 MOMENTUM BREAKOUT: RSI={rsi:.1f}, Vol={volume_ratio:.2f}x")
        
        # VENTA: Breakout bajista o sobrecomprado
        elif (rsi > 70 or  # Overbought
              (current_price > resistance * 0.98 and momentum < 0) or  # Cerca de resistencia con momentum negativo
              (macd < signal_line and rsi > 60)):  # MACD cruzando abajo
            
            signal = 'SELL'
            confidence = 0.7
            logger.info(f"⚠️ MOMENTUM REVERSAL: RSI={rsi:.1f}, Precio cerca resistencia")
        
        return signal, confidence
    
    @staticmethod
    def volume_surge(features: np.ndarray, market_context: dict) -> Tuple[str, float]:
        """
        Estrategia de Picos de Volumen
        
        SOL frecuentemente tiene picos de volumen que preceden movimientos grandes
        """
        volume_ratio = features[6] if len(features) > 6 else 1
        momentum = features[7] if len(features) > 7 else 0
        rsi = features[1] if len(features) > 1 else 50
        
        signal = 'HOLD'
        confidence = 0.5
        
        # Pico de volumen significativo (>2.5x normal)
        if volume_ratio > 2.5:
            if momentum > 0 and rsi < 60:
                # Volumen alcista
                signal = 'BUY'
                confidence = min(0.9, 0.7 + (volume_ratio - 2.5) * 0.05)
                logger.info(f"📊 VOLUME SURGE BUY: {volume_ratio:.1f}x volumen normal")
            
            elif momentum < 0 and rsi > 50:
                # Volumen bajista
                signal = 'SELL'
                confidence = 0.75
                logger.info(f"📊 VOLUME SURGE SELL: {volume_ratio:.1f}x volumen normal")
        
        return signal, confidence
    
    @staticmethod
    def bollinger_squeeze(features: np.ndarray, market_context: dict) -> Tuple[str, float]:
        """
        Estrategia de Compresión de Bollinger
        
        Cuando las Bandas de Bollinger se comprimen (baja volatilidad),
        usualmente precede un movimiento explosivo en SOL
        """
        bb_position = features[5] if len(features) > 5 else 0.5  # Posición en BB (0-1)
        volatility = market_context.get('volatility', 0)
        momentum = features[7] if len(features) > 7 else 0
        
        signal = 'HOLD'
        confidence = 0.5
        
        # Baja volatilidad (squeeze)
        if volatility < 0.02:  # Ajustar según datos históricos
            # Esperar breakout
            if bb_position < 0.2 and momentum > 0:
                # Bounce desde banda inferior con momentum
                signal = 'BUY'
                confidence = 0.75
                logger.info(f"🎯 BOLLINGER SQUEEZE BUY: Bounce desde banda inferior")
            
            elif bb_position > 0.8 and momentum < 0:
                # Rechazo desde banda superior
                signal = 'SELL'
                confidence = 0.7
                logger.info(f"🎯 BOLLINGER SQUEEZE SELL: Rechazo desde banda superior")
        
        return signal, confidence
    
    @staticmethod
    def support_resistance_scalp(features: np.ndarray, market_context: dict) -> Tuple[str, float]:
        """
        Estrategia de Scalping en Soporte/Resistencia
        
        SOL respeta bien niveles de S/R en timeframes cortos
        """
        current_price = market_context.get('current_price', 0)
        support = market_context.get('support', 0)
        resistance = market_context.get('resistance', float('inf'))
        rsi = features[1] if len(features) > 1 else 50
        
        signal = 'HOLD'
        confidence = 0.5
        
        # Cerca del soporte (dentro del 2%)
        if support > 0 and current_price <= support * 1.02:
            if rsi < 40:
                signal = 'BUY'
                # Más cerca = más confianza
                distance_pct = ((current_price - support) / support) * 100
                confidence = 0.8 - (distance_pct * 0.1)
                logger.info(f"🎯 SUPPORT BOUNCE: {distance_pct:.2f}% del soporte")
        
        # Cerca de la resistencia (dentro del 2%)
        elif resistance < float('inf') and current_price >= resistance * 0.98:
            if rsi > 60:
                signal = 'SELL'
                distance_pct = ((resistance - current_price) / resistance) * 100
                confidence = 0.75 - (distance_pct * 0.1)
                logger.info(f"🎯 RESISTANCE REJECTION: {distance_pct:.2f}% de la resistencia")
        
        return signal, confidence
    
    @staticmethod
    def trend_following(features: np.ndarray, market_context: dict) -> Tuple[str, float]:
        """
        Estrategia de Seguimiento de Tendencia
        
        SOL tiene tendencias fuertes - no pelear contra la tendencia
        """
        trend = market_context.get('trend', 'sideways')
        rsi = features[1] if len(features) > 1 else 50
        sma_distance = features[4] if len(features) > 4 else 0
        momentum = features[7] if len(features) > 7 else 0
        
        signal = 'HOLD'
        confidence = 0.5
        
        if trend == 'uptrend':
            # En tendencia alcista, comprar en pullbacks
            if rsi < 45 and momentum > -0.01:  # Pullback leve
                signal = 'BUY'
                confidence = 0.75
                logger.info(f"📈 TREND FOLLOWING: Pullback en uptrend")
        
        elif trend == 'downtrend':
            # En tendencia bajista, vender en rallies
            if rsi > 55 and momentum < 0.01:  # Rally leve
                signal = 'SELL'
                confidence = 0.7
                logger.info(f"📉 TREND FOLLOWING: Rally en downtrend")
        
        return signal, confidence


class StrategyEnsemble:
    """
    Combina múltiples estrategias para mejor performance
    """
    def __init__(self, strategies_weights: dict = None):
        if strategies_weights is None:
            # Pesos por defecto
            self.weights = {
                'momentum_breakout': 0.3,
                'volume_surge': 0.25,
                'bollinger_squeeze': 0.15,
                'support_resistance': 0.2,
                'trend_following': 0.1
            }
        else:
            self.weights = strategies_weights
        
        self.sol_strategies = SOLStrategies()
    
    def get_ensemble_signal(self, features: np.ndarray, market_context: dict, 
                           ml_prediction: np.ndarray) -> Tuple[str, float]:
        """
        Combinar todas las estrategias con sus pesos
        """
        # Obtener señales de cada estrategia
        signals = {}
        confidences = {}
        
        signals['momentum'], confidences['momentum'] = \
            self.sol_strategies.momentum_breakout(features, market_context)
        
        signals['volume'], confidences['volume'] = \
            self.sol_strategies.volume_surge(features, market_context)
        
        signals['bollinger'], confidences['bollinger'] = \
            self.sol_strategies.bollinger_squeeze(features, market_context)
        
        signals['support_resistance'], confidences['support_resistance'] = \
            self.sol_strategies.support_resistance_scalp(features, market_context)
        
        signals['trend'], confidences['trend'] = \
            self.sol_strategies.trend_following(features, market_context)
        
        # Mapear señales a valores numéricos
        signal_values = {'BUY': 1, 'HOLD': 0, 'SELL': -1}
        
        # Calcular señal ponderada
        weighted_signal = 0
        total_confidence = 0
        
        strategy_names = {
            'momentum': 'momentum_breakout',
            'volume': 'volume_surge',
            'bollinger': 'bollinger_squeeze',
            'support_resistance': 'support_resistance',
            'trend': 'trend_following'
        }
        
        for short_name, strategy_name in strategy_names.items():
            signal_val = signal_values[signals[short_name]]
            confidence = confidences[short_name]
            weight = self.weights[strategy_name]
            
            weighted_signal += signal_val * confidence * weight
            total_confidence += confidence * weight
        
        # Incorporar predicción ML (peso 0.3)
        ml_weight = 0.3
        ml_signal = ml_prediction[2] - ml_prediction[0]  # buy_prob - sell_prob
        weighted_signal += ml_signal * ml_weight
        total_confidence += ml_weight
        
        # Normalizar
        final_signal_value = weighted_signal / total_confidence
        final_confidence = abs(final_signal_value)
        
        # Determinar señal final
        if final_signal_value > 0.15:
            final_signal = 'BUY'
        elif final_signal_value < -0.15:
            final_signal = 'SELL'
        else:
            final_signal = 'HOLD'
        
        logger.info(f"🎲 ENSEMBLE: {final_signal} (confidence: {final_confidence:.2f}, value: {final_signal_value:.2f})")
        
        return final_signal, min(final_confidence, 0.95)


# Configuración recomendada para SOL
SOL_OPTIMAL_CONFIG = {
    "symbol": "SOL/USDT",
    "timeframe": "5m",  # SOL se mueve rápido, 5m es óptimo
    "stop_loss_pct": 0.025,  # 2.5% SL (SOL es volátil)
    "take_profit_pct": 0.055,  # 5.5% TP (ratio 2.2:1)
    "max_position_size": 0.90,  # 90% del capital
    "update_interval": 45,  # Check cada 45s
    
    # Estrategias habilitadas
    "use_ensemble": True,
    "strategy_weights": {
        "momentum_breakout": 0.30,
        "volume_surge": 0.25,
        "bollinger_squeeze": 0.15,
        "support_resistance": 0.20,
        "trend_following": 0.10
    }
}
