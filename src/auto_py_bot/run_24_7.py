#!/usr/bin/env python3
"""
Script para ejecutar el bot 24/7 con:
- Reinicio automático en caso de errores
- Monitoreo de salud
- Logs persistentes
- Manejo de interrupciones
"""

import subprocess
import sys
import time
import logging
from datetime import datetime
import signal
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot_supervisor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class BotSupervisor:
    """
    Supervisor que mantiene el bot ejecutándose 24/7
    """
    def __init__(self, max_restarts=10, restart_delay=60):
        self.max_restarts = max_restarts
        self.restart_delay = restart_delay
        self.restart_count = 0
        self.running = True
        self.process = None
        
        # Manejar señales de sistema
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """Manejar señales de interrupción"""
        logger.info(f"\nSeñal {signum} recibida, deteniendo bot...")
        self.running = False
        
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Proceso no respondió, forzando terminación...")
                self.process.kill()
        
        sys.exit(0)
    
    def check_dependencies(self):
        """Verificar que todas las dependencias estén instaladas"""
        logger.info("Verificando dependencias...")
        
        required_files = [
            'scripts/trading_bot.py',
            'src/auto_py_bot/ml_model.py',
            'src/auto_py_bot/data_collector.py',
            'src/auto_py_bot/risk_manager.py',
            'config.json'
        ]
        
        for file in required_files:
            if not os.path.exists(file):
                logger.error(f"Archivo requerido no encontrado: {file}")
                return False
        
        try:
            import torch
            import ccxt
            import pandas
            import numpy
            logger.info("✓ Todas las dependencias están instaladas")
            return True
        except ImportError as e:
            logger.error(f"Dependencia faltante: {e}")
            logger.error("Ejecuta: pip install -r requirements.txt")
            return False
    
    def run_bot(self):
        """Ejecutar el bot de trading"""
        logger.info("="*60)
        logger.info(f"INICIANDO BOT DE TRADING - {datetime.now()}")
        logger.info("="*60)
        
        try:
            self.process = subprocess.Popen(
                [sys.executable, 'scripts/trading_bot.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Leer output en tiempo real
            for line in self.process.stdout:
                print(line, end='')
                
            self.process.wait()
            return self.process.returncode
            
        except Exception as e:
            logger.error(f"Error ejecutando bot: {e}")
            return 1
    
    def start(self):
        """Iniciar supervisor"""
        if not self.check_dependencies():
            logger.error("No se puede iniciar - dependencias faltantes")
            return
        
        logger.info("Supervisor iniciado")
        logger.info(f"Reinicio automático: activado (max: {self.max_restarts})")
        logger.info(f"Delay entre reinicios: {self.restart_delay}s")
        
        while self.running and self.restart_count < self.max_restarts:
            logger.info(f"\nIntento #{self.restart_count + 1}")
            
            return_code = self.run_bot()
            
            if not self.running:
                break
            
            if return_code == 0:
                logger.info("Bot terminó normalmente")
                break
            else:
                logger.warning(f"Bot terminó con código {return_code}")
                self.restart_count += 1
                
                if self.restart_count < self.max_restarts:
                    logger.info(f"Reiniciando en {self.restart_delay}s...")
                    time.sleep(self.restart_delay)
                else:
                    logger.error(f"Máximo de reinicios ({self.max_restarts}) alcanzado")
                    break
        
        logger.info("Supervisor detenido")


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║         BOT DE TRADING ML - SUPERVISOR 24/7            ║
    ║                                                        ║
    ║  Este script mantendrá el bot ejecutándose            ║
    ║  continuamente con reinicio automático en caso        ║
    ║  de errores.                                          ║
    ║                                                        ║
    ║  Presiona Ctrl+C para detener el bot                  ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    supervisor = BotSupervisor(
        max_restarts=100,  # Reiniciar hasta 100 veces
        restart_delay=60    # Esperar 60s entre reinicios
    )
    
    supervisor.start()
