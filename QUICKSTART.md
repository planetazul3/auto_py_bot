# 🚀 GUÍA DE INICIO RÁPIDO - Bot Trading ML para SOL

Esta guía te llevará de 0 a tener el bot funcionando en **5 minutos**.

## ⚡ Setup Súper Rápido

### 1. Instalar Dependencias (1 minuto)

```bash
pip install -r requirements.txt
```

Si da error, instala uno por uno:
```bash
pip install torch numpy pandas scikit-learn ccxt tqdm matplotlib
```

### 2. Entrenar el Modelo (3-10 minutos)

**CRÍTICO**: El bot necesita un modelo entrenado para funcionar.

```bash
PYTHONPATH=src python scripts/train_model.py
```

Esto descargará datos de SOL y entrenará el modelo. Ve por un café ☕

### 3. ¡Ejecutar el Bot! (30 segundos)

**Versión básica** (sin extras):
```bash
PYTHONPATH=src python scripts/trading_bot.py
```

**Versión mejorada** (recomendada):
```bash
PYTHONPATH=src python scripts/enhanced_bot.py
```

**Versión 24/7** (con supervisor):
```bash
PYTHONPATH=src python scripts/run_24_7.py
```

## 🎯 ¿Qué Versión Usar?

### `trading_bot.py` - Básica
- ✅ Funcionalidad core
- ✅ Trading ML básico
- ❌ Sin Telegram
- ❌ Sin estrategias SOL

### `enhanced_bot.py` - Completa (RECOMENDADA)
- ✅ Todo lo de la básica
- ✅ Notificaciones Telegram
- ✅ Estrategias optimizadas para SOL
- ✅ Base de datos SQLite
- ✅ Ensemble de estrategias

### `run_24_7.py` - Producción
- Ejecuta cualquiera de las anteriores
- Reinicio automático si falla
- Logs persistentes

## 📱 Configurar Telegram (OPCIONAL - 2 minutos)

1. Abre Telegram y busca `@BotFather`
2. Envía `/newbot` y sigue instrucciones
3. Copia el **TOKEN** que te da
4. Busca `@userinfobot` y copia tu **CHAT_ID**
5. Edita `config.json`:

```json
{
    "use_telegram": true,
    "telegram_bot_token": "123456789:ABC-TU_TOKEN_AQUI",
    "telegram_chat_id": "987654321"
}
```

6. Reinicia el bot y ¡recibirás notificaciones!

## ⚙️ Configuración Mínima

El archivo `config.json` ya viene listo para usar. Solo verifica:

```json
{
    "paper_trading": true,    // true = simulado (SIN RIESGO)
    "initial_capital": 10.0,  // Capital simulado
    "symbol": "SOL/USDT"      // Par a tradear
}
```

## 🎮 Comandos Útiles

### Ver logs en tiempo real:
```bash
tail -f trading_bot.log
```

### Verificar salud del sistema:
```bash
PYTHONPATH=src python scripts/utils.py health
```

### Ver estadísticas rápidas:
```bash
PYTHONPATH=src python scripts/utils.py stats
```

### Hacer backtesting:
```bash
PYTHONPATH=src python scripts/backtest.py
```

## 🆘 Solución Rápida de Problemas

### "ModuleNotFoundError: No module named 'torch'"
```bash
pip install torch --break-system-packages
```

### "Error descargando datos"
- Verifica tu conexión a internet
- Binance puede estar caído, espera 5 min

### "trading_model.pth not found"
```bash
PYTHONPATH=src python scripts/train_model.py
```

### El bot no hace trades
- ¿Entrenaste el modelo? → `PYTHONPATH=src python scripts/train_model.py`
- ¿Está en paper trading? → Normal, es simulado
- ¿Hay suficiente capital? → Mínimo $5 en config

## 📊 Monitoreo Básico

El bot imprime estadísticas cada 10 iteraciones:

```
================================================================================
ESTADÍSTICAS DEL BOT
================================================================================
Capital actual: $10.45
ROI: 4.50%
Total trades: 23
Win rate: 65.22%
```

## 🔥 Mejores Prácticas

1. **SIEMPRE** empieza en `paper_trading: true`
2. **ENTRENA** el modelo antes de usar
3. **MONITOREA** las primeras horas
4. Deja correr **mínimo 24h** para ver resultados
5. Re-entrena el modelo cada **7-14 días**

## 🎓 Siguiente Nivel

Una vez que el bot esté funcionando:

1. **Optimiza parámetros** en `config.json`
2. **Activa Telegram** para notificaciones
3. **Haz backtesting** con diferentes configuraciones
4. **Revisa la base de datos** para analizar trades
5. Solo entonces considera **trading real** (con precaución)

## 💡 Tips Pro

- **Timeframe**: 5m es óptimo para SOL (rápido pero no demasiado)
- **Stop Loss**: 2.5% funciona bien para la volatilidad de SOL
- **Take Profit**: 5.5% da ratio risk/reward de 2.2:1
- **Update Interval**: 45s es un buen balance

## 🚨 Recuerda

- Paper trading = 0 riesgo
- Trading real = RIESGO de perder dinero
- Empieza con capital mínimo ($10-20)
- Nunca inviertas más de lo que puedes perder

---

**¿Listo? ¡Ejecuta el bot!**

```bash
PYTHONPATH=src python scripts/enhanced_bot.py
```

**¡Buena suerte! 🚀**
