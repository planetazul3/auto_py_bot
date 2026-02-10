#!/usr/bin/env python3
"""
Utilidades para monitoreo y análisis del bot
"""

import json
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_logs(log_file='trading_bot.log'):
    """
    Analizar logs del bot y extraer métricas
    """
    print("🔍 Analizando logs del bot...")
    
    with open(log_file, 'r') as f:
        logs = f.readlines()
    
    # Extraer información
    trades = []
    errors = []
    signals = []
    
    for line in logs:
        if 'Orden colocada' in line:
            trades.append(line)
        if 'ERROR' in line:
            errors.append(line)
        if 'Signal:' in line:
            signals.append(line)
    
    print(f"\n📊 Resumen:")
    print(f"  Total de trades ejecutados: {len(trades)}")
    print(f"  Total de señales generadas: {len(signals)}")
    print(f"  Total de errores: {len(errors)}")
    
    if errors:
        print(f"\n⚠️  Últimos errores:")
        for error in errors[-5:]:
            print(f"  {error.strip()}")
    
    return {
        'trades': len(trades),
        'signals': len(signals),
        'errors': len(errors)
    }


def check_model_performance(model_path='trading_model.pth'):
    """
    Verificar si el modelo existe y su antigüedad
    """
    import os
    from pathlib import Path
    
    print("\n🤖 Verificando modelo ML...")
    
    model_file = Path(model_path)
    
    if not model_file.exists():
        print("  ❌ Modelo no encontrado")
        print("  💡 Ejecuta: python train_model.py")
        return False
    
    # Antigüedad del modelo
    mod_time = datetime.fromtimestamp(model_file.stat().st_mtime)
    age = datetime.now() - mod_time
    
    print(f"  ✅ Modelo encontrado")
    print(f"  📅 Última actualización: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  ⏱️  Antigüedad: {age.days} días")
    
    if age.days > 14:
        print("  ⚠️  Modelo tiene más de 14 días")
        print("  💡 Considera re-entrenar: python train_model.py")
    
    return True


def check_config(config_file='config.json'):
    """
    Verificar configuración del bot
    """
    print("\n⚙️  Verificando configuración...")
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print(f"  ✅ Archivo de configuración válido")
        print(f"\n  Configuración actual:")
        print(f"    Exchange: {config.get('exchange')}")
        print(f"    Symbol: {config.get('symbol')}")
        print(f"    Timeframe: {config.get('timeframe')}")
        print(f"    Capital inicial: ${config.get('initial_capital')}")
        print(f"    Paper trading: {config.get('paper_trading')}")
        print(f"    Stop loss: {config.get('stop_loss_pct')*100}%")
        print(f"    Take profit: {config.get('take_profit_pct')*100}%")
        
        # Advertencias
        if not config.get('paper_trading'):
            print(f"\n  ⚠️  MODO TRADING REAL ACTIVADO")
            if not config.get('api_key') or not config.get('api_secret'):
                print(f"  ❌ API keys no configuradas!")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error leyendo configuración: {e}")
        return False


def monitor_capital():
    """
    Monitorear evolución del capital
    """
    print("\n💰 Monitoreando capital...")
    
    try:
        # Leer logs para extraer capital
        with open('trading_bot.log', 'r') as f:
            logs = f.readlines()
        
        capital_history = []
        
        for line in logs:
            if 'Capital:' in line:
                try:
                    # Extraer valor del capital
                    capital_str = line.split('Capital:')[1].split()[0].replace('$', '').replace(',', '')
                    capital = float(capital_str)
                    capital_history.append(capital)
                except:
                    continue
        
        if capital_history:
            current = capital_history[-1]
            initial = capital_history[0]
            roi = ((current - initial) / initial) * 100
            
            print(f"  Capital inicial: ${initial:.2f}")
            print(f"  Capital actual: ${current:.2f}")
            print(f"  ROI: {roi:.2f}%")
            
            # Gráfico simple
            if len(capital_history) > 5:
                plt.figure(figsize=(10, 4))
                plt.plot(capital_history, linewidth=2, color='green' if roi > 0 else 'red')
                plt.axhline(y=initial, color='gray', linestyle='--', alpha=0.5)
                plt.title('Evolución del Capital')
                plt.ylabel('Capital (USDT)')
                plt.xlabel('Actualizaciones')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig('capital_evolution.png', dpi=150)
                print(f"  📊 Gráfico guardado: capital_evolution.png")
        else:
            print(f"  ℹ️  No hay datos de capital aún")
            
    except Exception as e:
        print(f"  ⚠️  Error: {e}")


def system_health_check():
    """
    Verificación completa del sistema
    """
    print("="*60)
    print("🏥 VERIFICACIÓN DE SALUD DEL SISTEMA")
    print("="*60)
    
    checks = {
        'config': False,
        'model': False,
        'logs': False
    }
    
    # 1. Configuración
    checks['config'] = check_config()
    
    # 2. Modelo
    checks['model'] = check_model_performance()
    
    # 3. Logs
    try:
        analyze_logs()
        checks['logs'] = True
    except FileNotFoundError:
        print("\n📋 Logs no encontrados (el bot aún no se ha ejecutado)")
    
    # 4. Capital
    try:
        monitor_capital()
    except:
        pass
    
    # Resumen
    print("\n" + "="*60)
    print("📋 RESUMEN DE VERIFICACIÓN")
    print("="*60)
    
    all_ok = all(checks.values())
    
    for check, status in checks.items():
        symbol = "✅" if status else "❌"
        print(f"  {symbol} {check.capitalize()}")
    
    if all_ok:
        print("\n🎉 Todo listo para operar!")
    else:
        print("\n⚠️  Hay problemas que resolver antes de ejecutar el bot")
    
    return all_ok


def quick_stats():
    """
    Mostrar estadísticas rápidas del bot
    """
    print("\n📊 ESTADÍSTICAS RÁPIDAS")
    print("="*60)
    
    try:
        with open('trading_bot.log', 'r') as f:
            logs = f.readlines()
        
        # Última actualización
        if logs:
            last_line = logs[-1]
            print(f"Última actividad: {last_line[:19]}")
        
        # Buscar última estadística completa
        for line in reversed(logs):
            if 'Total trades:' in line:
                print(line.strip())
            if 'Win rate:' in line:
                print(line.strip())
            if 'Capital actual:' in line:
                print(line.strip())
                break
                
    except FileNotFoundError:
        print("No hay logs disponibles")


def main():
    """Función principal del script de utilidades"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'health':
            system_health_check()
        elif command == 'logs':
            analyze_logs()
        elif command == 'model':
            check_model_performance()
        elif command == 'config':
            check_config()
        elif command == 'capital':
            monitor_capital()
        elif command == 'stats':
            quick_stats()
        else:
            print(f"Comando desconocido: {command}")
            print("\nComandos disponibles:")
            print("  health  - Verificación completa del sistema")
            print("  logs    - Analizar logs del bot")
            print("  model   - Verificar modelo ML")
            print("  config  - Verificar configuración")
            print("  capital - Monitorear evolución del capital")
            print("  stats   - Estadísticas rápidas")
    else:
        # Por defecto, hacer health check completo
        system_health_check()


if __name__ == "__main__":
    main()
