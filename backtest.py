import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging

from ml_model import TradingModel, FeatureEngineer
from data_collector import DataCollector, MarketAnalyzer, SignalGenerator
from risk_manager import RiskManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Backtester:
    """
    Sistema de backtesting para probar estrategias con datos históricos
    """
    def __init__(self, initial_capital=10.0):
        self.initial_capital = initial_capital
        self.data_collector = DataCollector()
        self.ml_model = None
        
    def load_model(self, model_path='trading_model.pth'):
        """Cargar modelo entrenado"""
        try:
            self.ml_model = TradingModel(input_size=10)
            self.ml_model.load_model(model_path)
            logger.info("Modelo cargado exitosamente")
            return True
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            return False
    
    def run_backtest(self, symbol='SOL/USDT', timeframe='5m', days=7):
        """
        Ejecutar backtest completo
        """
        logger.info("="*70)
        logger.info("INICIANDO BACKTESTING")
        logger.info("="*70)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Timeframe: {timeframe}")
        logger.info(f"Period: {days} días")
        logger.info(f"Capital inicial: ${self.initial_capital}")
        
        # 1. Descargar datos históricos
        logger.info("\nDescargando datos históricos...")
        df = self.data_collector.get_historical_data(symbol, timeframe, limit=1000)
        
        if df is None or len(df) < 100:
            logger.error("Datos insuficientes")
            return None
        
        logger.info(f"Datos descargados: {len(df)} velas")
        logger.info(f"Período: {df.iloc[0]['timestamp']} a {df.iloc[-1]['timestamp']}")
        
        # 2. Preparar features
        logger.info("\nPreparando features...")
        candles = df.to_dict('records')
        features = FeatureEngineer.create_features(candles)
        
        # Normalizar
        features_normalized = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
        
        # 3. Inicializar componentes
        market_analyzer = MarketAnalyzer()
        signal_generator = SignalGenerator()
        risk_manager = RiskManager(initial_capital=self.initial_capital)
        
        # 4. Simular trading
        logger.info("\nSimulando trading...")
        sequence_length = 60
        equity_curve = [self.initial_capital]
        trades = []
        
        for i in range(sequence_length, len(df)):
            # Precio actual
            current_candle = df.iloc[i]
            current_price = float(current_candle['close'])
            current_volume = float(current_candle['volume'])
            timestamp = current_candle['timestamp']
            
            # Actualizar market analyzer
            market_analyzer.update(current_price, current_volume)
            
            # Crear secuencia para ML
            feature_sequence = features_normalized[i-sequence_length:i]
            
            # Predicción ML
            if self.ml_model:
                ml_prediction = self.ml_model.predict(feature_sequence)
            else:
                # Fallback: predicción aleatoria
                ml_prediction = np.random.dirichlet([1, 1, 1])
            
            # Contexto de mercado
            market_context = {
                'trend': market_analyzer.detect_trend(),
                'support': market_analyzer.calculate_support_resistance()[0],
                'resistance': market_analyzer.calculate_support_resistance()[1],
                'volatility': market_analyzer.calculate_volatility(),
                'current_price': current_price
            }
            
            # Generar señal
            signal, confidence = signal_generator.generate_signal(
                features[i],
                ml_prediction,
                market_context
            )
            
            # Verificar stop loss / take profit
            sl_tp = risk_manager.check_stop_loss_take_profit(current_price)
            if sl_tp:
                signal = 'SELL'
                confidence = 1.0
            
            # Ejecutar trade si procede
            can_trade = risk_manager.can_trade(signal, confidence, current_price)
            
            if can_trade:
                if signal == 'BUY':
                    position_size = risk_manager.calculate_position_size(current_price, confidence)
                    risk_manager.enter_position(current_price, position_size)
                    
                    trades.append({
                        'type': 'BUY',
                        'timestamp': timestamp,
                        'price': current_price,
                        'size': position_size,
                        'confidence': confidence
                    })
                    
                elif signal == 'SELL' and risk_manager.in_position:
                    pnl = risk_manager.exit_position(current_price)
                    
                    trades.append({
                        'type': 'SELL',
                        'timestamp': timestamp,
                        'price': current_price,
                        'size': risk_manager.position_size,
                        'pnl': pnl,
                        'confidence': confidence
                    })
            
            # Actualizar equity curve
            equity_curve.append(risk_manager.current_capital)
        
        # 5. Calcular métricas
        logger.info("\n" + "="*70)
        logger.info("RESULTADOS DEL BACKTEST")
        logger.info("="*70)
        
        stats = risk_manager.get_stats()
        
        # Métricas básicas
        logger.info(f"\nCapital Inicial: ${self.initial_capital:.2f}")
        logger.info(f"Capital Final: ${stats['current_capital']:.2f}")
        logger.info(f"ROI: {stats['roi']:.2f}%")
        logger.info(f"Total P&L: ${stats['total_pnl']:.4f}")
        
        # Trades
        logger.info(f"\nTotal Trades: {stats['total_trades']}")
        logger.info(f"Winning Trades: {stats['winning_trades']}")
        logger.info(f"Losing Trades: {stats['losing_trades']}")
        logger.info(f"Win Rate: {stats['win_rate']:.2f}%")
        
        # P&L por trade
        logger.info(f"\nAvg P&L por trade: ${stats['avg_pnl']:.4f}")
        logger.info(f"Avg Win: ${stats['avg_win']:.4f}")
        logger.info(f"Avg Loss: ${stats['avg_loss']:.4f}")
        
        # Risk metrics
        logger.info(f"\nMax Drawdown: {stats['max_drawdown']:.2f}%")
        logger.info(f"Total Fees: ${stats['total_fees']:.4f}")
        
        # Sharpe ratio
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
        logger.info(f"Sharpe Ratio: {sharpe:.2f}")
        
        # Profit factor
        total_wins = sum(t['pnl'] for t in risk_manager.trade_history if t['pnl'] > 0)
        total_losses = abs(sum(t['pnl'] for t in risk_manager.trade_history if t['pnl'] < 0))
        profit_factor = total_wins / (total_losses + 1e-10)
        logger.info(f"Profit Factor: {profit_factor:.2f}")
        
        logger.info("="*70)
        
        # 6. Visualizar resultados
        self.plot_results(df, equity_curve, trades, stats)
        
        return {
            'stats': stats,
            'equity_curve': equity_curve,
            'trades': trades,
            'sharpe_ratio': sharpe,
            'profit_factor': profit_factor
        }
    
    def plot_results(self, df, equity_curve, trades, stats):
        """
        Visualizar resultados del backtest
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # 1. Precio con trades
        ax1 = axes[0]
        prices = df['close'].values
        timestamps = range(len(df))
        
        ax1.plot(timestamps, prices, label='Precio SOL/USDT', color='blue', alpha=0.7)
        
        # Marcar trades
        buy_trades = [t for t in trades if t['type'] == 'BUY']
        sell_trades = [t for t in trades if t['type'] == 'SELL']
        
        if buy_trades:
            buy_indices = []
            buy_prices = []
            for t in buy_trades:
                # Encontrar índice aproximado
                idx = df[df['timestamp'] <= t['timestamp']].index[-1] if len(df[df['timestamp'] <= t['timestamp']]) > 0 else 0
                buy_indices.append(idx)
                buy_prices.append(t['price'])
            
            ax1.scatter(buy_indices, buy_prices, color='green', marker='^', s=100, label='BUY', zorder=5)
        
        if sell_trades:
            sell_indices = []
            sell_prices = []
            for t in sell_trades:
                idx = df[df['timestamp'] <= t['timestamp']].index[-1] if len(df[df['timestamp'] <= t['timestamp']]) > 0 else 0
                sell_indices.append(idx)
                sell_prices.append(t['price'])
            
            ax1.scatter(sell_indices, sell_prices, color='red', marker='v', s=100, label='SELL', zorder=5)
        
        ax1.set_title('Precio con Señales de Trading', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Precio (USDT)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Equity Curve
        ax2 = axes[1]
        ax2.plot(equity_curve, color='green', linewidth=2)
        ax2.axhline(y=self.initial_capital, color='gray', linestyle='--', label='Capital Inicial')
        ax2.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Capital (USDT)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Drawdown
        ax3 = axes[2]
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max * 100
        
        ax3.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
        ax3.plot(drawdown, color='red', linewidth=1)
        ax3.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Drawdown (%)')
        ax3.set_xlabel('Tiempo')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar gráfico
        filename = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"\nGráfico guardado: {filename}")
        
        try:
            plt.show()
        except:
            pass


def main():
    """Función principal"""
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║              BACKTESTING TRADING BOT ML                ║
    ║                                                        ║
    ║  Prueba tu estrategia con datos históricos            ║
    ║  sin arriesgar capital real                           ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    backtester = Backtester(initial_capital=10.0)
    
    # Cargar modelo si existe
    backtester.load_model('trading_model.pth')
    
    # Ejecutar backtest
    results = backtester.run_backtest(
        symbol='SOL/USDT',
        timeframe='5m',
        days=7
    )
    
    if results:
        logger.info("\n✅ Backtest completado exitosamente!")
        logger.info("📊 Revisa el gráfico generado para análisis visual")
    else:
        logger.error("\n❌ Backtest falló")


if __name__ == "__main__":
    main()
