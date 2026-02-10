import sqlite3
import pandas as pd
from datetime import datetime
import json
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class TradingDatabase:
    """
    Base de datos SQLite para almacenar histórico de trades y métricas
    """
    def __init__(self, db_path='trading_history.db'):
        self.db_path = db_path
        self.conn = None
        self.initialize_database()
    
    def initialize_database(self):
        """Crear tablas si no existen"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = self.conn.cursor()
            
            # Tabla de trades
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL,
                    exit_price REAL,
                    position_size REAL,
                    pnl REAL,
                    pnl_pct REAL,
                    fees REAL,
                    confidence REAL,
                    signal_type TEXT,
                    ml_prediction TEXT,
                    market_context TEXT,
                    capital_after REAL,
                    duration_minutes INTEGER
                )
            ''')
            
            # Tabla de métricas diarias
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT UNIQUE NOT NULL,
                    starting_capital REAL,
                    ending_capital REAL,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    total_pnl REAL,
                    total_fees REAL,
                    win_rate REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL
                )
            ''')
            
            # Tabla de señales generadas
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    confidence REAL,
                    price REAL,
                    executed BOOLEAN,
                    reason TEXT
                )
            ''')
            
            # Tabla de balance histórico
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS balance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    capital REAL,
                    in_position BOOLEAN,
                    position_value REAL
                )
            ''')
            
            # Tabla de configuración
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS bot_config (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    config_json TEXT NOT NULL
                )
            ''')
            
            self.conn.commit()
            logger.info(f"✅ Base de datos inicializada: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Error inicializando base de datos: {e}")
    
    def save_trade(self, trade_data: dict):
        """Guardar trade en la base de datos"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
                INSERT INTO trades (
                    timestamp, symbol, side, entry_price, exit_price,
                    position_size, pnl, pnl_pct, fees, confidence,
                    signal_type, ml_prediction, market_context, capital_after, duration_minutes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data.get('timestamp', datetime.now().isoformat()),
                trade_data.get('symbol', 'SOL/USDT'),
                trade_data.get('side', 'BUY'),
                trade_data.get('entry_price'),
                trade_data.get('exit_price'),
                trade_data.get('position_size'),
                trade_data.get('pnl'),
                trade_data.get('pnl_pct'),
                trade_data.get('fees'),
                trade_data.get('confidence'),
                trade_data.get('signal_type'),
                json.dumps(trade_data.get('ml_prediction', [])),
                json.dumps(trade_data.get('market_context', {})),
                trade_data.get('capital_after'),
                trade_data.get('duration_minutes')
            ))
            
            self.conn.commit()
            logger.info(f"💾 Trade guardado en DB: {trade_data.get('side')} @ ${trade_data.get('entry_price', 0):.4f}")
            
        except Exception as e:
            logger.error(f"Error guardando trade: {e}")
    
    def save_signal(self, signal_data: dict):
        """Guardar señal generada"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
                INSERT INTO signals (timestamp, symbol, signal, confidence, price, executed, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal_data.get('timestamp', datetime.now().isoformat()),
                signal_data.get('symbol', 'SOL/USDT'),
                signal_data.get('signal'),
                signal_data.get('confidence'),
                signal_data.get('price'),
                signal_data.get('executed', False),
                signal_data.get('reason', '')
            ))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Error guardando señal: {e}")
    
    def save_balance_snapshot(self, capital: float, in_position: bool, position_value: float = 0):
        """Guardar snapshot del balance"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
                INSERT INTO balance_history (timestamp, capital, in_position, position_value)
                VALUES (?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                capital,
                in_position,
                position_value
            ))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Error guardando balance: {e}")
    
    def save_daily_metrics(self, metrics: dict):
        """Guardar métricas diarias"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO daily_metrics (
                    date, starting_capital, ending_capital, total_trades,
                    winning_trades, losing_trades, total_pnl, total_fees,
                    win_rate, sharpe_ratio, max_drawdown
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.get('date', datetime.now().date().isoformat()),
                metrics.get('starting_capital'),
                metrics.get('ending_capital'),
                metrics.get('total_trades', 0),
                metrics.get('winning_trades', 0),
                metrics.get('losing_trades', 0),
                metrics.get('total_pnl', 0),
                metrics.get('total_fees', 0),
                metrics.get('win_rate', 0),
                metrics.get('sharpe_ratio', 0),
                metrics.get('max_drawdown', 0)
            ))
            
            self.conn.commit()
            logger.info(f"💾 Métricas diarias guardadas")
            
        except Exception as e:
            logger.error(f"Error guardando métricas diarias: {e}")
    
    def get_all_trades(self, limit: int = 100) -> pd.DataFrame:
        """Obtener todos los trades"""
        try:
            query = f"SELECT * FROM trades ORDER BY timestamp DESC LIMIT {limit}"
            df = pd.read_sql_query(query, self.conn)
            return df
        except Exception as e:
            logger.error(f"Error obteniendo trades: {e}")
            return pd.DataFrame()
    
    def get_trades_by_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Obtener trades en un rango de fechas"""
        try:
            query = '''
                SELECT * FROM trades 
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            '''
            df = pd.read_sql_query(query, self.conn, params=(start_date, end_date))
            return df
        except Exception as e:
            logger.error(f"Error obteniendo trades por fecha: {e}")
            return pd.DataFrame()
    
    def get_winning_trades(self) -> pd.DataFrame:
        """Obtener solo trades ganadores"""
        try:
            query = "SELECT * FROM trades WHERE pnl > 0 ORDER BY pnl DESC"
            df = pd.read_sql_query(query, self.conn)
            return df
        except Exception as e:
            logger.error(f"Error obteniendo trades ganadores: {e}")
            return pd.DataFrame()
    
    def get_losing_trades(self) -> pd.DataFrame:
        """Obtener solo trades perdedores"""
        try:
            query = "SELECT * FROM trades WHERE pnl < 0 ORDER BY pnl ASC"
            df = pd.read_sql_query(query, self.conn)
            return df
        except Exception as e:
            logger.error(f"Error obteniendo trades perdedores: {e}")
            return pd.DataFrame()
    
    def get_daily_metrics(self, days: int = 30) -> pd.DataFrame:
        """Obtener métricas diarias"""
        try:
            query = f"SELECT * FROM daily_metrics ORDER BY date DESC LIMIT {days}"
            df = pd.read_sql_query(query, self.conn)
            return df
        except Exception as e:
            logger.error(f"Error obteniendo métricas diarias: {e}")
            return pd.DataFrame()
    
    def get_balance_history(self) -> pd.DataFrame:
        """Obtener historial de balance"""
        try:
            query = "SELECT * FROM balance_history ORDER BY timestamp"
            df = pd.read_sql_query(query, self.conn)
            return df
        except Exception as e:
            logger.error(f"Error obteniendo historial de balance: {e}")
            return pd.DataFrame()
    
    def get_statistics(self) -> dict:
        """Obtener estadísticas generales"""
        try:
            cursor = self.conn.cursor()
            
            # Total trades
            cursor.execute("SELECT COUNT(*) FROM trades")
            total_trades = cursor.fetchone()[0]
            
            # Win rate
            cursor.execute("SELECT COUNT(*) FROM trades WHERE pnl > 0")
            winning = cursor.fetchone()[0]
            win_rate = (winning / total_trades * 100) if total_trades > 0 else 0
            
            # Total P&L
            cursor.execute("SELECT SUM(pnl) FROM trades")
            total_pnl = cursor.fetchone()[0] or 0
            
            # Total fees
            cursor.execute("SELECT SUM(fees) FROM trades")
            total_fees = cursor.fetchone()[0] or 0
            
            # Mejor trade
            cursor.execute("SELECT MAX(pnl) FROM trades")
            best_trade = cursor.fetchone()[0] or 0
            
            # Peor trade
            cursor.execute("SELECT MIN(pnl) FROM trades")
            worst_trade = cursor.fetchone()[0] or 0
            
            # Promedio de ganancia
            cursor.execute("SELECT AVG(pnl) FROM trades WHERE pnl > 0")
            avg_win = cursor.fetchone()[0] or 0
            
            # Promedio de pérdida
            cursor.execute("SELECT AVG(pnl) FROM trades WHERE pnl < 0")
            avg_loss = cursor.fetchone()[0] or 0
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning,
                'losing_trades': total_trades - winning,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'total_fees': total_fees,
                'net_pnl': total_pnl - total_fees,
                'best_trade': best_trade,
                'worst_trade': worst_trade,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {}
    
    def export_to_csv(self, output_dir='.'):
        """Exportar todas las tablas a CSV"""
        try:
            # Trades
            trades_df = self.get_all_trades(limit=10000)
            trades_df.to_csv(f'{output_dir}/trades_history.csv', index=False)
            
            # Daily metrics
            metrics_df = self.get_daily_metrics(days=365)
            metrics_df.to_csv(f'{output_dir}/daily_metrics.csv', index=False)
            
            # Balance history
            balance_df = self.get_balance_history()
            balance_df.to_csv(f'{output_dir}/balance_history.csv', index=False)
            
            logger.info(f"✅ Datos exportados a CSV en {output_dir}/")
            
        except Exception as e:
            logger.error(f"Error exportando a CSV: {e}")
    
    def close(self):
        """Cerrar conexión a la base de datos"""
        if self.conn:
            self.conn.close()
            logger.info("Base de datos cerrada")


def generate_report(db: TradingDatabase):
    """Generar reporte completo del histórico"""
    print("\n" + "="*70)
    print("📊 REPORTE DE TRADING - HISTÓRICO COMPLETO")
    print("="*70)
    
    stats = db.get_statistics()
    
    print(f"\n📈 ESTADÍSTICAS GENERALES:")
    print(f"  Total de trades: {stats['total_trades']}")
    print(f"  Trades ganadores: {stats['winning_trades']}")
    print(f"  Trades perdedores: {stats['losing_trades']}")
    print(f"  Win Rate: {stats['win_rate']:.2f}%")
    
    print(f"\n💰 P&L:")
    print(f"  Total P&L: ${stats['total_pnl']:.4f}")
    print(f"  Total Fees: ${stats['total_fees']:.4f}")
    print(f"  P&L Neto: ${stats['net_pnl']:.4f}")
    
    print(f"\n🎯 TRADES:")
    print(f"  Mejor trade: ${stats['best_trade']:.4f}")
    print(f"  Peor trade: ${stats['worst_trade']:.4f}")
    print(f"  Ganancia promedio: ${stats['avg_win']:.4f}")
    print(f"  Pérdida promedio: ${stats['avg_loss']:.4f}")
    print(f"  Profit Factor: {stats['profit_factor']:.2f}")
    
    print("="*70)


if __name__ == "__main__":
    # Ejemplo de uso
    db = TradingDatabase()
    generate_report(db)
    db.close()
