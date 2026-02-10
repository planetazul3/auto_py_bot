# 🤖 Bot de Trading Automatizado con ML para Criptomonedas

Bot de trading automatizado 24/7 que usa **PyTorch** y **Machine Learning** para operar en SOL/USDT con solo **$10 USD** de capital inicial.

## 🌟 Características

- **Machine Learning Avanzado**: Red LSTM con mecanismo de atención
- **Indicadores Técnicos**: RSI, MACD, Bandas de Bollinger, volatilidad
- **Gestión de Riesgo Inteligente**: Kelly Criterion, stop loss, take profit
- **24/7 Automatizado**: Supervisor con reinicio automático
- **Paper Trading**: Modo simulado para pruebas sin riesgo
- **Multi-Exchange**: Soporta Binance, Kraken, Coinbase y más (vía CCXT)

## 📊 Estrategias Implementadas

El bot combina múltiples estrategias que han demostrado funcionar:

1. **Predicción ML**: LSTM entrenado con datos históricos
2. **Análisis de Tendencias**: Detección automática de uptrend/downtrend
3. **Soporte/Resistencia**: Identificación de niveles clave
4. **Volume Analysis**: Detección de picos de volumen
5. **Risk Management**: Kelly Criterion para sizing óptimo

## 🚀 Instalación Rápida

### Requisitos
- Python 3.8+
- GPU (opcional, mejora velocidad)

### Paso 1: Clonar/Descargar archivos

```bash
# Todos los archivos deben estar en el mismo directorio
ls
# Deberías ver:
# src/auto_py_bot/*.py, scripts/*.py, config.json, requirements.txt
```

### Paso 2: Instalar dependencias

```bash
pip install -r requirements.txt
```

### Paso 3: Configurar el bot

Edita `config.json`:

```json
{
    "exchange": "binance",
    "symbol": "SOL/USDT",
    "initial_capital": 10.0,
    "paper_trading": true,  // true = simulado, false = real
    "api_key": null,        // Solo si paper_trading = false
    "api_secret": null      // Solo si paper_trading = false
}
```

### Paso 4: Entrenar el modelo (IMPORTANTE)

```bash
PYTHONPATH=src python scripts/train_model.py
```

Esto descargará 30 días de datos históricos y entrenará el modelo ML.
Tardará ~10-30 minutos dependiendo de tu hardware.

### Paso 5: Ejecutar el bot

#### Opción A: Ejecución simple
```bash
PYTHONPATH=src python scripts/trading_bot.py
```

#### Opción B: Ejecución 24/7 con supervisor (RECOMENDADO)
```bash
PYTHONPATH=src python scripts/run_24_7.py
```

El supervisor reiniciará automáticamente el bot si hay algún error.

## 📈 Uso

### Modo Paper Trading (Simulado)

Por defecto, el bot opera en modo simulado (sin riesgo):

```json
{
    "paper_trading": true,
    "initial_capital": 10.0
}
```

Esto te permite:
- ✅ Probar estrategias sin riesgo
- ✅ Ver cómo funciona el bot
- ✅ Entrenar y optimizar el modelo

### Modo Trading Real

⚠️ **ADVERTENCIA**: Trading real implica riesgo de pérdida de capital

1. Crear API keys en tu exchange (Binance recomendado)
2. Actualizar `config.json`:

```json
{
    "paper_trading": false,
    "api_key": "TU_API_KEY_AQUI",
    "api_secret": "TU_API_SECRET_AQUI"
}
```

3. **RECOMENDACIÓN**: Empezar con capital mínimo ($10-20)

## ⚙️ Configuración Avanzada

### Parámetros de Risk Management

```json
{
    "max_position_size": 0.95,      // 95% del capital máximo por trade
    "max_daily_loss": 0.15,         // Detener si pérdida diaria > 15%
    "max_drawdown": 0.30,           // Detener si drawdown > 30%
    "stop_loss_pct": 0.03,          // Stop loss a 3%
    "take_profit_pct": 0.06         // Take profit a 6%
}
```

### Timeframes Disponibles

```json
{
    "timeframe": "5m"   // Opciones: "1m", "5m", "15m", "1h", "4h"
}
```

⚠️ Timeframes más cortos (1m, 5m) = más trades, más fees
✅ Timeframes más largos (1h, 4h) = menos trades, menos fees

## 🎯 Estrategias Descubiertas por Usuarios

### 1. **Scalping en Volatilidad** (5m timeframe)
- Aprovecha movimientos rápidos
- Stop loss ajustado (2-3%)
- Take profit pequeño (4-6%)

### 2. **Swing Trading** (1h-4h timeframe)
- Sigue tendencias más largas
- Stop loss más amplio (5-8%)
- Take profit mayor (10-15%)

### 3. **Mean Reversion** 
- Compra en soporte
- Vende en resistencia
- Usa Bandas de Bollinger

## 📊 Monitoreo

El bot genera logs en:
- `trading_bot.log` - Log principal del bot
- `bot_supervisor.log` - Log del supervisor 24/7

### Ver estadísticas en tiempo real:

```bash
tail -f trading_bot.log
```

Verás:
```
================================================================================
ESTADÍSTICAS DEL BOT
================================================================================
Capital actual: $10.45
ROI: 4.50%
Total trades: 23
Win rate: 65.22%
Total P&L: $0.4500
Max drawdown: 8.20%
Sharpe Ratio: 1.45
================================================================================
```

## 🔧 Mantenimiento

### Re-entrenar el modelo

Recomendado cada 7-14 días para adaptarse a nuevas condiciones:

```bash
PYTHONPATH=src python scripts/train_model.py
```

### Optimizar parámetros

1. Ejecuta el bot por 7 días en paper trading
2. Analiza `trading_bot.log` y `performance_report.txt`
3. Ajusta parámetros en `config.json` según resultados
4. Re-entrena el modelo si es necesario

## 🛡️ Gestión de Riesgo

### Reglas de Oro:

1. **NUNCA** inviertas más de lo que puedes perder
2. **SIEMPRE** empieza en paper trading
3. **NUNCA** desactives el stop loss
4. **MONITOREA** diariamente durante los primeros 7 días
5. **RETIRA** ganancias regularmente

### Límites de Seguridad Integrados:

- ✅ Stop loss automático
- ✅ Take profit automático
- ✅ Límite de pérdida diaria
- ✅ Límite de drawdown máximo
- ✅ Tamaño de posición calculado con Kelly Criterion

## 📱 Ejecución en Servidor/VPS

Para ejecutar 24/7 en un servidor:

### Con screen:
```bash
screen -S trading_bot
PYTHONPATH=src python scripts/run_24_7.py
# Presiona Ctrl+A luego D para detach
```

Para reconectar:
```bash
screen -r trading_bot
```

### Con systemd (Linux):

Crear `/etc/systemd/system/trading-bot.service`:

```ini
[Unit]
Description=Trading Bot ML
After=network.target

[Service]
Type=simple
User=tu_usuario
WorkingDirectory=/ruta/al/bot
ExecStart=/usr/bin/python3 run_24_7.py
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
```

Luego:
```bash
sudo systemctl enable trading-bot
sudo systemctl start trading-bot
sudo systemctl status trading-bot
```

## 🐛 Troubleshooting

### Error: "No se pudieron descargar datos"
- Verifica conexión a Internet
- Verifica que el exchange esté funcionando
- Intenta otro exchange en config.json

### Error: "API key inválida"
- Verifica que las keys sean correctas
- Asegúrate de habilitar trading en las API keys
- Verifica que la IP esté permitida (whitelist)

### Error: "Modelo no encontrado"
- Ejecuta `PYTHONPATH=src python scripts/train_model.py` primero

### Bot no ejecuta trades
- Verifica que `paper_trading` esté configurado correctamente
- Revisa que el capital sea > $5
- Asegúrate de que el modelo esté entrenado

## 📈 Resultados Esperados

Con $10 USD inicial y configuración por defecto:

- **Win Rate**: 55-65% (típico para bots ML)
- **ROI Mensual**: 5-20% (depende de volatilidad)
- **Sharpe Ratio**: 1.0-2.0 (bueno)
- **Max Drawdown**: 10-25%

⚠️ **IMPORTANTE**: Resultados pasados NO garantizan resultados futuros.

## 🔒 Seguridad

1. **NUNCA** compartas tus API keys
2. **USA** API keys con solo permisos de trading (NO withdrawal)
3. **ACTIVA** IP whitelist en el exchange
4. **GUARDA** backups de `trading_model.pth` regularmente

## 🤝 Contribuciones

Ideas para mejorar:

- [ ] Integración con Telegram para notificaciones
- [ ] Dashboard web en tiempo real
- [ ] Soporte para más criptomonedas simultáneamente
- [ ] Backtesting más avanzado
- [ ] Auto-optimización de hiperparámetros

## 📄 Licencia

Este proyecto es de código abierto. Úsalo bajo tu propio riesgo.

## ⚠️ Disclaimer

Este bot es para fines educativos. El trading de criptomonedas es altamente riesgoso y puede resultar en pérdida total del capital. El autor NO se hace responsable por pérdidas financieras. Siempre haz tu propia investigación (DYOR).

---

**¿Preguntas?** Revisa los logs en `trading_bot.log` primero.

**¡Happy Trading! 🚀**


## 🗂️ Nueva Estructura Profesional

```text
auto_py_bot/
├── src/auto_py_bot/   # Código fuente principal
├── scripts/           # Puntos de entrada ejecutables
├── docs/              # Documentación técnica
├── tests/             # Pruebas automatizadas
├── deploy/            # Archivos de despliegue
├── config.json
├── requirements.txt
└── Makefile
```

Comandos recomendados:

```bash
make test
make run
make supervise
```
