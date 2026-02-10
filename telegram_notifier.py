import requests
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Sistema de notificaciones vía Telegram
    """
    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = bool(bot_token and chat_id)
        
        if self.enabled:
            logger.info("✅ Notificaciones de Telegram habilitadas")
        else:
            logger.info("ℹ️  Notificaciones de Telegram deshabilitadas (configura bot_token y chat_id)")
    
    def send_message(self, message: str, parse_mode: str = "HTML"):
        """Enviar mensaje a Telegram"""
        if not self.enabled:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                return True
            else:
                logger.error(f"Error enviando mensaje a Telegram: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Excepción enviando mensaje a Telegram: {e}")
            return False
    
    def notify_trade(self, trade_type: str, symbol: str, price: float, 
                     size: float, confidence: float):
        """Notificar trade ejecutado"""
        emoji = "🟢" if trade_type == "BUY" else "🔴"
        
        message = f"""
{emoji} <b>{trade_type} EJECUTADO</b>

💱 Par: <code>{symbol}</code>
💰 Precio: <code>${price:.4f}</code>
📊 Cantidad: <code>{size:.6f}</code>
🎯 Confianza: <code>{confidence*100:.1f}%</code>

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return self.send_message(message)
    
    def notify_pnl(self, pnl: float, pnl_pct: float, capital: float, 
                   entry_price: float, exit_price: float):
        """Notificar resultado del trade"""
        emoji = "💚" if pnl > 0 else "❤️"
        result = "GANANCIA" if pnl > 0 else "PÉRDIDA"
        
        message = f"""
{emoji} <b>{result}</b>

💵 P&L: <code>${pnl:.4f}</code> ({pnl_pct:.2f}%)
💰 Capital: <code>${capital:.2f}</code>

📈 Entrada: <code>${entry_price:.4f}</code>
📉 Salida: <code>${exit_price:.4f}</code>

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return self.send_message(message)
    
    def notify_stop_loss(self, price: float, loss_pct: float):
        """Notificar activación de stop loss"""
        message = f"""
🛑 <b>STOP LOSS ACTIVADO</b>

💰 Precio: <code>${price:.4f}</code>
📉 Pérdida: <code>{loss_pct:.2f}%</code>

⚠️ Posición cerrada automáticamente

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return self.send_message(message)
    
    def notify_take_profit(self, price: float, gain_pct: float):
        """Notificar activación de take profit"""
        message = f"""
🎯 <b>TAKE PROFIT ACTIVADO</b>

💰 Precio: <code>${price:.4f}</code>
📈 Ganancia: <code>{gain_pct:.2f}%</code>

✅ Objetivo alcanzado

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return self.send_message(message)
    
    def notify_error(self, error_msg: str):
        """Notificar error crítico"""
        message = f"""
⚠️ <b>ERROR CRÍTICO</b>

{error_msg}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return self.send_message(message)
    
    def notify_daily_summary(self, stats: dict):
        """Notificar resumen diario"""
        roi_emoji = "📈" if stats['roi'] > 0 else "📉"
        
        message = f"""
📊 <b>RESUMEN DIARIO</b>

{roi_emoji} ROI: <code>{stats['roi']:.2f}%</code>
💰 Capital: <code>${stats['current_capital']:.2f}</code>
💵 P&L: <code>${stats['total_pnl']:.4f}</code>

📈 Trades totales: <code>{stats['total_trades']}</code>
✅ Ganadores: <code>{stats.get('winning_trades', 0)}</code>
❌ Perdedores: <code>{stats.get('losing_trades', 0)}</code>
🎯 Win Rate: <code>{stats['win_rate']:.1f}%</code>

📊 Sharpe Ratio: <code>{stats.get('sharpe_ratio', 0):.2f}</code>
📉 Max DD: <code>{stats['max_drawdown']:.2f}%</code>

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return self.send_message(message)
    
    def notify_bot_start(self, config: dict):
        """Notificar inicio del bot"""
        mode = "SIMULADO 🧪" if config.get('paper_trading') else "REAL 💰"
        
        message = f"""
🤖 <b>BOT INICIADO</b>

🎮 Modo: <code>{mode}</code>
💱 Par: <code>{config.get('symbol')}</code>
⏱️ Timeframe: <code>{config.get('timeframe')}</code>
💰 Capital: <code>${config.get('initial_capital'):.2f}</code>

🛡️ Stop Loss: <code>{config.get('stop_loss_pct')*100}%</code>
🎯 Take Profit: <code>{config.get('take_profit_pct')*100}%</code>

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return self.send_message(message)
    
    def notify_bot_stop(self, final_stats: dict):
        """Notificar detención del bot"""
        message = f"""
🛑 <b>BOT DETENIDO</b>

📊 Estadísticas finales:
💰 Capital final: <code>${final_stats['current_capital']:.2f}</code>
📈 ROI: <code>{final_stats['roi']:.2f}%</code>
📊 Total trades: <code>{final_stats['total_trades']}</code>
🎯 Win rate: <code>{final_stats['win_rate']:.1f}%</code>

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return self.send_message(message)


# Función para obtener bot token y chat ID
def setup_telegram_bot():
    """
    Guía para configurar el bot de Telegram
    """
    print("""
╔════════════════════════════════════════════════════════════╗
║          CONFIGURACIÓN DE NOTIFICACIONES TELEGRAM          ║
╚════════════════════════════════════════════════════════════╝

Para recibir notificaciones en Telegram:

1. Abre Telegram y busca @BotFather
2. Envía /newbot y sigue las instrucciones
3. Copia el TOKEN que te da (ej: 123456789:ABCdefGHIjklMNOpqrsTUVwxyz)
4. Busca @userinfobot en Telegram
5. Inicia conversación y copia tu CHAT_ID (ej: 987654321)

Luego agrega en config.json:

{
    ...
    "telegram_bot_token": "TU_TOKEN_AQUI",
    "telegram_chat_id": "TU_CHAT_ID_AQUI"
}

Reinicia el bot y recibirás notificaciones!
""")


if __name__ == "__main__":
    setup_telegram_bot()
