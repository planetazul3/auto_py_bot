import torch
import numpy as np
import time
import logging
from datetime import datetime
import json
import sys
from pathlib import Path

from .ml_model import TradingModel, FeatureEngineer
from .data_collector import DataCollector, MarketAnalyzer, SignalGenerator
from .risk_manager import RiskManager, PerformanceTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TradingBot:
    """
    Bot de trading automatizado con ML
    """
    def __init__(self, config_file='config.json'):
        logger.info("=" * 60)
        logger.info("INICIALIZANDO BOT DE TRADING ML")
        logger.info("=" * 60)
        
        # Cargar configuración
        self.config = self.load_config(config_file)
        
        # Inicializar componentes
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
        
        # Estado del bot
        self.symbol = self.config['symbol']
        self.timeframe = self.config['timeframe']
        self.running = False
        self.sequence_length = 60  # Usar últimas 60 velas para predicción
        self.update_interval = self.config['update_interval']
        
        # Cargar modelo si existe
        model_path = Path('trading_model.pth')
        if model_path.exists():
            try:
                self.ml_model.load_model(str(model_path))
                logger.info("Modelo ML cargado exitosamente")
            except Exception as e:
                logger.warning(f"No se pudo cargar el modelo: {e}")
        
        logger.info(f"Bot configurado para {self.symbol} en timeframe {self.timeframe}")
        logger.info(f"Capital inicial: ${self.config['initial_capital']}")
        
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
        """Configuración por defecto"""
        return {
            'exchange': 'binance',
            'symbol': 'SOL/USDT',
            'timeframe': '5m',
            'initial_capital': 10.0,
            'max_position_size': 0.95,
            'max_daily_loss': 0.15,
            'stop_loss_pct': 0.03,
            'take_profit_pct': 0.06,
            'update_interval': 60,  # segundos
            'api_key': None,
            'api_secret': None
        }
    
    def prepare_features(self, df):
        """
        Preparar features para el modelo ML
        """
        candles = df.to_dict('records')
        features = FeatureEngineer.create_features(candles)
        
        return features
    
    def get_market_context(self):
        """
        Obtener contexto actual del mercado
        """
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
    
    def execute_trade(self, signal, confidence):
        """
        Ejecutar trade basado en señal
        """
        current_price = self.data_collector.get_current_price(self.symbol)
        
        if signal == 'BUY':
            # Calcular tamaño de posición
            position_size = self.risk_manager.calculate_position_size(current_price, confidence)
            
            # Intentar ejecutar orden
            try:
                # En modo PAPER TRADING (simulado)
                if self.config.get('paper_trading', True):
                    logger.info(f"[PAPER] Comprando {position_size:.6f} SOL @ ${current_price:.2f}")
                    self.risk_manager.enter_position(current_price, position_size)
                else:
                    # Trading real
                    order = self.data_collector.place_order(
                        self.symbol, 'market', 'buy', position_size
                    )
                    if order:
                        actual_price = float(order['price'])
                        actual_size = float(order['amount'])
                        self.risk_manager.enter_position(actual_price, actual_size)
                        
            except Exception as e:
                logger.error(f"Error ejecutando compra: {e}")
        
        elif signal == 'SELL' and self.risk_manager.in_position:
            # Vender posición completa
            try:
                if self.config.get('paper_trading', True):
                    logger.info(f"[PAPER] Vendiendo {self.risk_manager.position_size:.6f} SOL @ ${current_price:.2f}")
                    self.risk_manager.exit_position(current_price)
                else:
                    # Trading real
                    order = self.data_collector.place_order(
                        self.symbol, 'market', 'sell', self.risk_manager.position_size
                    )
                    if order:
                        actual_price = float(order['price'])
                        self.risk_manager.exit_position(actual_price)
                        
            except Exception as e:
                logger.error(f"Error ejecutando venta: {e}")
    
    def trading_loop(self):
        """
        Loop principal de trading
        """
        logger.info("Iniciando loop de trading...")
        
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
                
                # 2. Actualizar analyzer con precio y volumen actual
                current_price = float(df.iloc[-1]['close'])
                current_volume = float(df.iloc[-1]['volume'])
                self.market_analyzer.update(current_price, current_volume)
                
                # 3. Preparar features para ML
                features = self.prepare_features(df)
                
                if len(features) < self.sequence_length:
                    logger.warning("Features insuficientes")
                    time.sleep(self.update_interval)
                    continue
                
                # Tomar últimas N features como secuencia
                feature_sequence = features[-self.sequence_length:]
                
                # Normalizar
                feature_sequence = (feature_sequence - np.mean(feature_sequence, axis=0)) / (np.std(feature_sequence, axis=0) + 1e-8)
                
                # 4. Obtener predicción del modelo ML
                ml_prediction = self.ml_model.predict(feature_sequence)
                
                # 5. Obtener contexto de mercado
                market_context = self.get_market_context()
                
                logger.info(f"Precio: ${current_price:.2f} | Trend: {market_context['trend']} | Volatilidad: {market_context['volatility']:.4f}")
                logger.info(f"ML Prediction - BUY: {ml_prediction[0]:.2f} | HOLD: {ml_prediction[1]:.2f} | SELL: {ml_prediction[2]:.2f}")
                
                # 6. Generar señal de trading
                signal, confidence = self.signal_generator.generate_signal(
                    features[-1],
                    ml_prediction,
                    market_context
                )
                
                # 7. Verificar stop loss / take profit
                if self.risk_manager.in_position:
                    sl_tp_signal = self.risk_manager.check_stop_loss_take_profit(current_price)
                    if sl_tp_signal:
                        signal = 'SELL'
                        confidence = 1.0
                
                # 8. Verificar si podemos operar
                can_trade = self.risk_manager.can_trade(signal, confidence, current_price)
                
                # 9. Ejecutar trade si procede
                if can_trade and signal in ['BUY', 'SELL']:
                    self.execute_trade(signal, confidence)
                
                # 10. Actualizar tracking
                self.performance_tracker.update(self.risk_manager.current_capital)
                
                # 11. Mostrar estadísticas cada 10 iteraciones
                if iteration % 10 == 0:
                    self.print_stats()
                
                # 12. Esperar antes de siguiente iteración
                logger.info(f"Esperando {self.update_interval}s hasta próxima actualización...")
                time.sleep(self.update_interval)
                
            except KeyboardInterrupt:
                logger.info("\nInterrupción detectada, deteniendo bot...")
                self.stop()
                break
                
            except Exception as e:
                logger.error(f"Error en trading loop: {e}", exc_info=True)
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
        logger.info("Bot iniciado!")
        self.trading_loop()
    
    def stop(self):
        """Detener el bot"""
        self.running = False
        logger.info("Bot detenido")
        
        # Guardar estadísticas finales
        self.print_stats()
        self.performance_tracker.save_performance_report()
        
        # Guardar modelo
        self.ml_model.save_model()
        logger.info("Modelo guardado")


if __name__ == "__main__":
    # Crear y ejecutar bot
    bot = TradingBot('config.json')
    
    try:
        bot.start()
    except KeyboardInterrupt:
        logger.info("\nDeteniendo bot...")
        bot.stop()
