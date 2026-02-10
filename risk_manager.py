import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Gestor de riesgo para el trading bot
    Implementa múltiples estrategias de protección de capital
    """
    def __init__(self, 
                 initial_capital=10.0,
                 max_position_size=0.95,  # 95% del capital máximo por trade
                 max_daily_loss=0.15,      # 15% pérdida máxima diaria
                 max_drawdown=0.30,        # 30% drawdown máximo
                 min_confidence=0.6,       # Confianza mínima para operar
                 stop_loss_pct=0.03,       # 3% stop loss
                 take_profit_pct=0.06):    # 6% take profit
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.min_confidence = min_confidence
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Tracking
        self.peak_capital = initial_capital
        self.daily_start_capital = initial_capital
        self.trade_history = []
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0)
        
        # Estado
        self.in_position = False
        self.position_entry_price = 0
        self.position_size = 0
        self.consecutive_losses = 0
        
    def reset_daily_stats(self):
        """Reset estadísticas diarias"""
        now = datetime.now()
        if now >= self.daily_reset_time + timedelta(days=1):
            self.daily_start_capital = self.current_capital
            self.daily_reset_time = now.replace(hour=0, minute=0, second=0)
            logger.info(f"Stats diarias reseteadas. Capital: ${self.current_capital:.2f}")
    
    def can_trade(self, signal, confidence, current_price):
        """
        Verificar si se puede ejecutar un trade
        """
        self.reset_daily_stats()
        
        # Check 1: Confianza mínima
        if confidence < self.min_confidence:
            logger.info(f"Trade rechazado: Confianza {confidence:.2f} < {self.min_confidence}")
            return False
        
        # Check 2: Pérdida diaria máxima
        daily_loss = (self.daily_start_capital - self.current_capital) / self.daily_start_capital
        if daily_loss > self.max_daily_loss:
            logger.warning(f"Límite de pérdida diaria alcanzado: {daily_loss*100:.2f}%")
            return False
        
        # Check 3: Drawdown máximo
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if drawdown > self.max_drawdown:
            logger.warning(f"Drawdown máximo alcanzado: {drawdown*100:.2f}%")
            return False
        
        # Check 4: Ya estamos en posición?
        if self.in_position and signal == 'BUY':
            logger.info("Ya estamos en posición, no se puede comprar más")
            return False
        
        if not self.in_position and signal == 'SELL':
            logger.info("No hay posición para vender")
            return False
        
        # Check 5: Consecutive losses (reducir tamaño)
        if self.consecutive_losses >= 3:
            logger.warning(f"3+ pérdidas consecutivas, reduciendo agresividad")
        
        return True
    
    def calculate_position_size(self, current_price, confidence):
        """
        Calcular tamaño de posición usando Kelly Criterion modificado
        """
        # Kelly Criterion: f = (p * b - q) / b
        # donde p = probabilidad de ganar, q = probabilidad de perder, b = ratio ganancia/pérdida
        
        win_prob = confidence
        loss_prob = 1 - confidence
        win_loss_ratio = self.take_profit_pct / self.stop_loss_pct
        
        kelly_fraction = (win_prob * win_loss_ratio - loss_prob) / win_loss_ratio
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25% (quarter Kelly)
        
        # Ajustar por consecutive losses
        if self.consecutive_losses >= 3:
            kelly_fraction *= 0.5
        
        # Calcular cantidad
        position_value = self.current_capital * kelly_fraction * self.max_position_size
        
        # Fees (0.1% típico en Binance)
        fee_factor = 0.999
        position_size = (position_value * fee_factor) / current_price
        
        logger.info(f"Tamaño de posición calculado: {position_size:.6f} SOL (${position_value:.2f})")
        
        return position_size
    
    def enter_position(self, entry_price, position_size):
        """Registrar entrada en posición"""
        self.in_position = True
        self.position_entry_price = entry_price
        self.position_size = position_size
        
        logger.info(f"Entrando en posición: {position_size:.6f} SOL @ ${entry_price:.2f}")
    
    def exit_position(self, exit_price):
        """Registrar salida de posición y calcular P&L"""
        if not self.in_position:
            return 0
        
        # Calcular profit/loss
        pnl_per_unit = exit_price - self.position_entry_price
        total_pnl = pnl_per_unit * self.position_size
        
        # Fees (0.1% en cada lado)
        entry_fee = self.position_entry_price * self.position_size * 0.001
        exit_fee = exit_price * self.position_size * 0.001
        total_fees = entry_fee + exit_fee
        
        net_pnl = total_pnl - total_fees
        pnl_pct = net_pnl / (self.position_entry_price * self.position_size)
        
        # Actualizar capital
        self.current_capital += net_pnl
        
        # Actualizar peak
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        # Track consecutive losses
        if net_pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Registrar trade
        trade_record = {
            'entry_time': datetime.now() - timedelta(minutes=5),  # Aproximado
            'exit_time': datetime.now(),
            'entry_price': self.position_entry_price,
            'exit_price': exit_price,
            'position_size': self.position_size,
            'pnl': net_pnl,
            'pnl_pct': pnl_pct * 100,
            'fees': total_fees,
            'capital_after': self.current_capital
        }
        self.trade_history.append(trade_record)
        
        logger.info(f"Saliendo de posición: {self.position_size:.6f} SOL @ ${exit_price:.2f}")
        logger.info(f"P&L: ${net_pnl:.4f} ({pnl_pct*100:.2f}%) | Capital: ${self.current_capital:.2f}")
        
        # Reset position
        self.in_position = False
        self.position_entry_price = 0
        self.position_size = 0
        
        return net_pnl
    
    def check_stop_loss_take_profit(self, current_price):
        """
        Verificar si se activa stop loss o take profit
        """
        if not self.in_position:
            return None
        
        price_change = (current_price - self.position_entry_price) / self.position_entry_price
        
        # Stop Loss
        if price_change <= -self.stop_loss_pct:
            logger.warning(f"STOP LOSS activado! Pérdida: {price_change*100:.2f}%")
            return 'STOP_LOSS'
        
        # Take Profit
        if price_change >= self.take_profit_pct:
            logger.info(f"TAKE PROFIT activado! Ganancia: {price_change*100:.2f}%")
            return 'TAKE_PROFIT'
        
        return None
    
    def get_stats(self):
        """Obtener estadísticas del trading"""
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'current_capital': self.current_capital,
                'roi': 0
            }
        
        total_trades = len(self.trade_history)
        winning_trades = [t for t in self.trade_history if t['pnl'] > 0]
        
        stats = {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': total_trades - len(winning_trades),
            'win_rate': len(winning_trades) / total_trades * 100,
            'total_pnl': sum(t['pnl'] for t in self.trade_history),
            'total_fees': sum(t['fees'] for t in self.trade_history),
            'avg_pnl': np.mean([t['pnl'] for t in self.trade_history]),
            'avg_win': np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t['pnl'] for t in self.trade_history if t['pnl'] < 0]) if any(t['pnl'] < 0 for t in self.trade_history) else 0,
            'current_capital': self.current_capital,
            'roi': (self.current_capital - self.initial_capital) / self.initial_capital * 100,
            'max_drawdown': (self.peak_capital - self.current_capital) / self.peak_capital * 100,
            'consecutive_losses': self.consecutive_losses
        }
        
        return stats


class PerformanceTracker:
    """
    Tracker de rendimiento del bot
    """
    def __init__(self):
        self.metrics = {
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'total_return': 0
        }
        self.equity_curve = []
        
    def update(self, capital, timestamp=None):
        """Actualizar equity curve"""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'capital': capital
        })
    
    def calculate_sharpe_ratio(self, risk_free_rate=0.02):
        """Calcular Sharpe Ratio"""
        if len(self.equity_curve) < 2:
            return 0
        
        returns = []
        for i in range(1, len(self.equity_curve)):
            prev_capital = self.equity_curve[i-1]['capital']
            curr_capital = self.equity_curve[i]['capital']
            ret = (curr_capital - prev_capital) / prev_capital
            returns.append(ret)
        
        if not returns:
            return 0
        
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        sharpe = (avg_return - risk_free_rate/365) / std_return
        
        return sharpe
    
    def save_performance_report(self, filename='performance_report.txt'):
        """Guardar reporte de rendimiento"""
        with open(filename, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("REPORTE DE RENDIMIENTO DEL BOT\n")
            f.write("=" * 50 + "\n\n")
            
            for key, value in self.metrics.items():
                f.write(f"{key}: {value}\n")
            
            f.write(f"\nSharpe Ratio: {self.calculate_sharpe_ratio():.2f}\n")
