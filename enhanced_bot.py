import torch
import numpy as np
import time
import logging
from datetime import datetime, timedelta
import json
import sys
from pathlib import Path

from ml_model import TradingModel, FeatureEngineer
from data_collector import DataCollector, MarketAnalyzer, SignalGenerator
from risk_manager import RiskManager, PerformanceTracker
from telegram_notifier import TelegramNotifier
from sol_strategies import StrategyEnsemble, SOL_OPTIMAL_CONFIG
from trading_database import TradingDatabase

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class EnhancedTradingBot:
    """
    Bot de trading mejorado con:
    - Notificaciones Telegram
    - Estrategias específicas para SOL
    - Base de datos para histórico
    - Ensemble de estrategias
    """
    def __init__(self, config_file='config.json'):
        logger.info("=" * 60)
        logger.info("INICIALIZANDO BOT DE TRADING ML MEJORADO")
        logger.info("=" * 60)
        
        # Cargar configuración
        self.config = self.load_config(config_file)
        
        # Inicializar componentes base
        self.data_collector = DataCollector(
            exchange_name=self.config['exchange'],
            api_key=self.config.get('api_key'),
            api_secret=self.config.get('api_secret')
        )
        
        self.ml_model = TradingModel(input_size=10)
        self.market_analyzer = MarketAnalyzer()
        self.signal_generator = SignalGenerator()
        
        self.risk_manager = RiskManager(
            initial_capital=self.config['initial_capital'],
            max_position_size=self.config['max_position_size'],
            max_daily_loss=self.config['max_daily_loss'],
            stop_loss_pct=self.config['stop_loss_pct'],
            take_profit_pct=self.config['take_profit_pct']
        )
        
        self.performance_tracker = PerformanceTracker()
        
        # Telegram notifier
        self.telegram = TelegramNotifier(
            bot_token=self.config.get('telegram_bot_token'),
            chat_id=self.config.get('telegram_chat_id')
        ) if self.config.get('use_telegram') else None
        
        # Database
        self.db = TradingDatabase(
            db_path=self.config.get('database_path', 'trading_history.db')
        ) if self.config.get('use_database', True) else None
        
        # SOL Strategies Ensemble
        self.use_sol_strategies = self.config.get('use_sol_strategies', True)
        if self.use_sol_strategies and self.config.get('use_ensemble', True):
            self.strategy_ensemble = StrategyEnsemble(
                strategies_weights=self.config.get('strategy_weights')
            )
            logger.info("✅ Estrategias SOL Ensemble activadas")
        else:
            self.strategy_ensemble = None
        
        # Estado del bot
        self.symbol = self.config['symbol']
        self.timeframe = self.config['timeframe']
        self.running = False
        self.sequence_length = 60
        self.update_interval = self.config['update_interval']
        self.last_daily_summary = None
        
        # Cargar modelo si existe
        model_path = Path('trading_model.pth')
        if model_path.exists():
            try:
                self.ml_model.load_model(str(model_path))
                logger.info("✅ Modelo ML cargado exitosamente")
            except Exception as e:
                logger.warning(f"⚠️  No se pudo cargar el modelo: {e}")
        
        logger.info(f"Bot configurado para {self.symbol} en timeframe {self.timeframe}")
        logger.info(f"Capital inicial: ${self.config['initial_capital']}")
        
        # Enviar notificación de inicio
        if self.telegram and self.telegram.enabled:
            self.telegram.notify_bot_start(self.config)
    
    def load_config(self, config_file):
        """Cargar configuración desde archivo JSON"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuración cargada desde {config_file}")
            return config
        except FileNotFoundError:
            logger.warning(f"Archivo de configuración no encontrado, usando defaults")
            return self.get_default_config()
    
    def get_default_config(self):
        """Configuración por defecto optimizada para SOL"""
        return SOL_OPTIMAL_CONFIG.copy()
    
    def prepare_features(self, df):
        """Preparar features para el modelo ML"""
        candles = df.to_dict('records')
        features = FeatureEngineer.create_features(candles)
        return features
    
    def get_market_context(self):
        """Obtener contexto actual del mercado"""
        trend = self.market_analyzer.detect_trend()
        support, resistance = self.market_analyzer.calculate_support_resistance()
        volatility = self.market_analyzer.calculate_volatility()
        volume_spike = self.market_analyzer.detect_volume_spike()
        current_price = self.data_collector.get_current_price(self.symbol)
        
        context = {
            'trend': trend,
            'support': support,
            'resistance': resistance,
            'volatility': volatility,
            'volume_spike': volume_spike,
            'current_price': current_price,
            'timestamp': datetime.now()
        }
        
        return context
    
    def execute_trade(self, signal, confidence, market_context, ml_prediction):
        """Ejecutar trade basado en señal"""
        current_price = self.data_collector.get_current_price(self.symbol)
        
        if signal == 'BUY':
            position_size = self.risk_manager.calculate_position_size(current_price, confidence)
            
            try:
                if self.config.get('paper_trading', True):
                    logger.info(f"[PAPER] Comprando {position_size:.6f} SOL @ ${current_price:.2f}")
                    self.risk_manager.enter_position(current_price, position_size)
                    
                    # Notificar
                    if self.telegram and self.telegram.enabled:
                        self.telegram.notify_trade('BUY', self.symbol, current_price, position_size, confidence)
                    
                    # Guardar en DB
                    if self.db:
                        self.db.save_trade({
                            'timestamp': datetime.now().isoformat(),
                            'symbol': self.symbol,
                            'side': 'BUY',
                            'entry_price': current_price,
                            'position_size': position_size,
                            'confidence': confidence,
                            'ml_prediction': ml_prediction.tolist() if hasattr(ml_prediction, 'tolist') else ml_prediction,
                            'market_context': market_context,
                            'signal_type': 'ensemble' if self.strategy_ensemble else 'ml'
                        })
                else:
                    # Trading real
                    order = self.data_collector.place_order(
                        self.symbol, 'market', 'buy', position_size
                    )
                    if order:
                        actual_price = float(order['price'])
                        actual_size = float(order['amount'])
                        self.risk_manager.enter_position(actual_price, actual_size)
                        
                        if self.telegram and self.telegram.enabled:
                            self.telegram.notify_trade('BUY', self.symbol, actual_price, actual_size, confidence)
                        
            except Exception as e:
                logger.error(f"Error ejecutando compra: {e}")
                if self.telegram and self.telegram.enabled:
                    self.telegram.notify_error(f"Error en compra: {str(e)}")
        
        elif signal == 'SELL' and self.risk_manager.in_position:
            try:
                entry_price = self.risk_manager.position_entry_price
                position_size = self.risk_manager.position_size
                
                if self.config.get('paper_trading', True):
                    logger.info(f"[PAPER] Vendiendo {position_size:.6f} SOL @ ${current_price:.2f}")
                    pnl = self.risk_manager.exit_position(current_price)
                    
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    
                    # Notificar
                    if self.telegram and self.telegram.enabled:
                        self.telegram.notify_pnl(pnl, pnl_pct, 
                                                 self.risk_manager.current_capital,
                                                 entry_price, current_price)
                    
                    # Guardar en DB
                    if self.db:
                        self.db.save_trade({
                            'timestamp': datetime.now().isoformat(),
                            'symbol': self.symbol,
                            'side': 'SELL',
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'position_size': position_size,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'confidence': confidence,
                            'capital_after': self.risk_manager.current_capital
                        })
                else:
                    # Trading real
                    order = self.data_collector.place_order(
                        self.symbol, 'market', 'sell', position_size
                    )
                    if order:
                        actual_price = float(order['price'])
                        pnl = self.risk_manager.exit_position(actual_price)
                        
                        if self.telegram and self.telegram.enabled:
                            pnl_pct = ((actual_price - entry_price) / entry_price) * 100
                            self.telegram.notify_pnl(pnl, pnl_pct,
                                                     self.risk_manager.current_capital,
                                                     entry_price, actual_price)
                        
            except Exception as e:
                logger.error(f"Error ejecutando venta: {e}")
                if self.telegram and self.telegram.enabled:
                    self.telegram.notify_error(f"Error en venta: {str(e)}")
    
    def check_daily_summary(self):
        """Enviar resumen diario si corresponde"""
        if not self.config.get('send_daily_summary', True):
            return
        
        if not self.telegram or not self.telegram.enabled:
            return
        
        now = datetime.now()
        summary_time = self.config.get('daily_summary_time', '23:00')
        summary_hour, summary_minute = map(int, summary_time.split(':'))
        
        # Verificar si es hora de enviar y no se ha enviado hoy
        if (now.hour == summary_hour and now.minute >= summary_minute and
            (self.last_daily_summary is None or 
             self.last_daily_summary.date() < now.date())):
            
            stats = self.risk_manager.get_stats()
            stats['sharpe_ratio'] = self.performance_tracker.calculate_sharpe_ratio()
            
            self.telegram.notify_daily_summary(stats)
            self.last_daily_summary = now
            
            # Guardar métricas diarias en DB
            if self.db:
                self.db.save_daily_metrics({
                    'date': now.date().isoformat(),
                    'starting_capital': self.risk_manager.daily_start_capital,
                    'ending_capital': self.risk_manager.current_capital,
                    'total_trades': stats['total_trades'],
                    'winning_trades': stats.get('winning_trades', 0),
                    'losing_trades': stats.get('losing_trades', 0),
                    'total_pnl': stats['total_pnl'],
                    'total_fees': stats.get('total_fees', 0),
                    'win_rate': stats['win_rate'],
                    'sharpe_ratio': stats['sharpe_ratio'],
                    'max_drawdown': stats['max_drawdown']
                })
    
    def trading_loop(self):
        """Loop principal de trading mejorado"""
        logger.info("Iniciando loop de trading mejorado...")
        
        iteration = 0
        
        while self.running:
            try:
                iteration += 1
                logger.info(f"\n{'='*60}")
                logger.info(f"ITERACIÓN #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"{'='*60}")
                
                # 1. Obtener datos históricos
                df = self.data_collector.get_historical_data(
                    self.symbol, 
                    self.timeframe, 
                    limit=500
                )
                
                if df is None or len(df) < self.sequence_length:
                    logger.warning("No hay suficientes datos, esperando...")
                    time.sleep(self.update_interval)
                    continue
                
                # 2. Actualizar analyzer
                current_price = float(df.iloc[-1]['close'])
                current_volume = float(df.iloc[-1]['volume'])
                self.market_analyzer.update(current_price, current_volume)
                
                # 3. Preparar features
                features = self.prepare_features(df)
                
                if len(features) < self.sequence_length:
                    logger.warning("Features insuficientes")
                    time.sleep(self.update_interval)
                    continue
                
                feature_sequence = features[-self.sequence_length:]
                feature_sequence = (feature_sequence - np.mean(feature_sequence, axis=0)) / (np.std(feature_sequence, axis=0) + 1e-8)
                
                # 4. Predicción ML
                ml_prediction = self.ml_model.predict(feature_sequence)
                
                # 5. Contexto de mercado
                market_context = self.get_market_context()
                
                logger.info(f"Precio: ${current_price:.2f} | Trend: {market_context['trend']} | Vol: {market_context['volatility']:.4f}")
                logger.info(f"ML - BUY: {ml_prediction[0]:.2f} | HOLD: {ml_prediction[1]:.2f} | SELL: {ml_prediction[2]:.2f}")
                
                # 6. Generar señal con estrategias SOL si está habilitado
                if self.strategy_ensemble:
                    signal, confidence = self.strategy_ensemble.get_ensemble_signal(
                        features[-1],
                        market_context,
                        ml_prediction
                    )
                else:
                    signal, confidence = self.signal_generator.generate_signal(
                        features[-1],
                        ml_prediction,
                        market_context
                    )
                
                # Guardar señal en DB
                if self.db:
                    self.db.save_signal({
                        'timestamp': datetime.now().isoformat(),
                        'symbol': self.symbol,
                        'signal': signal,
                        'confidence': confidence,
                        'price': current_price,
                        'executed': False
                    })
                
                # 7. Verificar stop loss / take profit
                if self.risk_manager.in_position:
                    sl_tp_signal = self.risk_manager.check_stop_loss_take_profit(current_price)
                    if sl_tp_signal:
                        signal = 'SELL'
                        confidence = 1.0
                        
                        # Notificar
                        if self.telegram and self.telegram.enabled:
                            if sl_tp_signal == 'STOP_LOSS':
                                loss_pct = ((current_price - self.risk_manager.position_entry_price) / 
                                           self.risk_manager.position_entry_price) * 100
                                self.telegram.notify_stop_loss(current_price, loss_pct)
                            elif sl_tp_signal == 'TAKE_PROFIT':
                                gain_pct = ((current_price - self.risk_manager.position_entry_price) / 
                                           self.risk_manager.position_entry_price) * 100
                                self.telegram.notify_take_profit(current_price, gain_pct)
                
                # 8. Verificar si podemos operar
                can_trade = self.risk_manager.can_trade(signal, confidence, current_price)
                
                # 9. Ejecutar trade
                if can_trade and signal in ['BUY', 'SELL']:
                    self.execute_trade(signal, confidence, market_context, ml_prediction)
                
                # 10. Actualizar tracking
                self.performance_tracker.update(self.risk_manager.current_capital)
                
                # Guardar snapshot de balance
                if self.db and iteration % 5 == 0:  # Cada 5 iteraciones
                    self.db.save_balance_snapshot(
                        self.risk_manager.current_capital,
                        self.risk_manager.in_position,
                        self.risk_manager.position_size * current_price if self.risk_manager.in_position else 0
                    )
                
                # 11. Estadísticas cada 10 iteraciones
                if iteration % 10 == 0:
                    self.print_stats()
                
                # 12. Verificar resumen diario
                self.check_daily_summary()
                
                # 13. Esperar
                logger.info(f"Esperando {self.update_interval}s hasta próxima actualización...")
                time.sleep(self.update_interval)
                
            except KeyboardInterrupt:
                logger.info("\nInterrupción detectada, deteniendo bot...")
                self.stop()
                break
                
            except Exception as e:
                logger.error(f"Error en trading loop: {e}", exc_info=True)
                if self.telegram and self.telegram.enabled:
                    self.telegram.notify_error(f"Error en loop: {str(e)}")
                time.sleep(self.update_interval)
    
    def print_stats(self):
        """Imprimir estadísticas del bot"""
        stats = self.risk_manager.get_stats()
        
        logger.info("\n" + "=" * 60)
        logger.info("ESTADÍSTICAS DEL BOT")
        logger.info("=" * 60)
        logger.info(f"Capital actual: ${stats['current_capital']:.2f}")
        logger.info(f"ROI: {stats['roi']:.2f}%")
        logger.info(f"Total trades: {stats['total_trades']}")
        logger.info(f"Win rate: {stats['win_rate']:.2f}%")
        logger.info(f"Total P&L: ${stats['total_pnl']:.4f}")
        logger.info(f"Avg P&L: ${stats['avg_pnl']:.4f}")
        logger.info(f"Max drawdown: {stats['max_drawdown']:.2f}%")
        logger.info(f"Sharpe Ratio: {self.performance_tracker.calculate_sharpe_ratio():.2f}")
        logger.info("=" * 60 + "\n")
    
    def start(self):
        """Iniciar el bot"""
        self.running = True
        logger.info("🚀 Bot iniciado!")
        self.trading_loop()
    
    def stop(self):
        """Detener el bot"""
        self.running = False
        logger.info("Bot detenido")
        
        # Estadísticas finales
        self.print_stats()
        stats = self.risk_manager.get_stats()
        
        # Notificar
        if self.telegram and self.telegram.enabled:
            stats['sharpe_ratio'] = self.performance_tracker.calculate_sharpe_ratio()
            self.telegram.notify_bot_stop(stats)
        
        # Guardar modelo
        self.ml_model.save_model()
        logger.info("Modelo guardado")
        
        # Cerrar DB
        if self.db:
            self.db.close()


if __name__ == "__main__":
    bot = EnhancedTradingBot('config.json')
    
    try:
        bot.start()
    except KeyboardInterrupt:
        logger.info("\nDeteniendo bot...")
        bot.stop()
