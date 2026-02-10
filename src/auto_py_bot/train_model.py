import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from tqdm import tqdm
import pickle

from .ml_model import TradingModel, FeatureEngineer
from .data_collector import DataCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Entrenador del modelo de ML
    """
    def __init__(self, symbol='SOL/USDT', timeframe='5m'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.data_collector = DataCollector()
        self.scaler = StandardScaler()
        
    def download_training_data(self, days=30):
        """
        Descargar datos históricos para entrenamiento
        """
        logger.info(f"Descargando {days} días de datos de {self.symbol}...")
        
        all_data = []
        
        # Binance limita a 1000 velas por request
        # 5min * 1000 = ~3.5 días
        requests_needed = (days * 24 * 60) // (5 * 1000) + 1
        
        for i in range(requests_needed):
            df = self.data_collector.get_historical_data(
                self.symbol,
                self.timeframe,
                limit=1000
            )
            
            if df is not None:
                all_data.append(df)
            
        if not all_data:
            logger.error("No se pudieron descargar datos")
            return None
        
        # Combinar todos los datos
        full_df = pd.concat(all_data, ignore_index=True)
        full_df = full_df.drop_duplicates(subset=['timestamp'])
        full_df = full_df.sort_values('timestamp')
        
        logger.info(f"Total de velas descargadas: {len(full_df)}")
        
        return full_df
    
    def create_labels(self, df, lookahead=5, threshold=0.005):
        """
        Crear labels para entrenamiento
        
        Label 0: SELL (precio baja más de threshold%)
        Label 1: HOLD (precio se mantiene)
        Label 2: BUY (precio sube más de threshold%)
        """
        labels = []
        
        for i in range(len(df) - lookahead):
            current_price = float(df.iloc[i]['close'])
            future_price = float(df.iloc[i + lookahead]['close'])
            
            price_change = (future_price - current_price) / current_price
            
            if price_change > threshold:
                labels.append(2)  # BUY
            elif price_change < -threshold:
                labels.append(0)  # SELL
            else:
                labels.append(1)  # HOLD
        
        # Últimas N filas no tienen label (no hay futuro)
        labels.extend([1] * lookahead)
        
        return np.array(labels)
    
    def prepare_sequences(self, features, labels, sequence_length=60):
        """
        Preparar secuencias para LSTM
        """
        X, y = [], []
        
        for i in range(len(features) - sequence_length):
            X.append(features[i:i+sequence_length])
            y.append(labels[i+sequence_length])
        
        return np.array(X), np.array(y)
    
    def train_model(self, epochs=50, batch_size=32, validation_split=0.2):
        """
        Entrenar el modelo completo
        """
        logger.info("="*60)
        logger.info("INICIANDO ENTRENAMIENTO DEL MODELO")
        logger.info("="*60)
        
        # 1. Descargar datos
        df = self.download_training_data(days=30)
        
        if df is None or len(df) < 500:
            logger.error("Datos insuficientes para entrenamiento")
            return
        
        # 2. Crear features
        logger.info("Creando features técnicas...")
        candles = df.to_dict('records')
        features = FeatureEngineer.create_features(candles)
        
        # 3. Crear labels
        logger.info("Creando labels...")
        labels = self.create_labels(df, lookahead=5, threshold=0.005)
        
        # 4. Normalizar features
        logger.info("Normalizando features...")
        features_scaled = self.scaler.fit_transform(features)
        
        # Guardar scaler
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # 5. Crear secuencias
        logger.info("Creando secuencias temporales...")
        X, y = self.prepare_sequences(features_scaled, labels, sequence_length=60)
        
        logger.info(f"Forma de X: {X.shape}")
        logger.info(f"Forma de y: {y.shape}")
        
        # Distribución de clases
        unique, counts = np.unique(y, return_counts=True)
        logger.info("Distribución de clases:")
        for label, count in zip(unique, counts):
            label_name = ['SELL', 'HOLD', 'BUY'][int(label)]
            logger.info(f"  {label_name}: {count} ({count/len(y)*100:.1f}%)")
        
        # 6. Split train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, shuffle=False
        )
        
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        
        # 7. Inicializar modelo
        model = TradingModel(input_size=features.shape[1])
        
        # 8. Entrenar
        logger.info(f"\nEntrenando por {epochs} epochs...")
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            model.model.train()
            train_losses = []
            
            # Mini-batches
            n_batches = len(X_train) // batch_size
            
            with tqdm(total=n_batches, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for i in range(0, len(X_train) - batch_size, batch_size):
                    batch_X = X_train[i:i+batch_size]
                    batch_y = y_train[i:i+batch_size]
                    
                    loss = model.train_step(batch_X, batch_y)
                    train_losses.append(loss)
                    
                    pbar.update(1)
                    pbar.set_postfix({'loss': f"{loss:.4f}"})
            
            avg_train_loss = np.mean(train_losses)
            
            # Validation
            model.model.eval()
            val_losses = []
            correct = 0
            total = 0
            
            with torch.no_grad():
                for i in range(0, len(X_val) - batch_size, batch_size):
                    batch_X = X_val[i:i+batch_size]
                    batch_y = y_val[i:i+batch_size]
                    
                    batch_X_tensor = torch.FloatTensor(batch_X).to(model.device)
                    batch_y_tensor = torch.LongTensor(batch_y).to(model.device)
                    
                    outputs = model.model(batch_X_tensor)
                    loss = model.criterion(outputs, batch_y_tensor)
                    val_losses.append(loss.item())
                    
                    # Accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y_tensor.size(0)
                    correct += (predicted == batch_y_tensor).sum().item()
            
            avg_val_loss = np.mean(val_losses)
            val_accuracy = 100 * correct / total
            
            logger.info(f"Epoch {epoch+1}/{epochs} - "
                       f"Train Loss: {avg_train_loss:.4f} - "
                       f"Val Loss: {avg_val_loss:.4f} - "
                       f"Val Acc: {val_accuracy:.2f}%")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                model.save_model('trading_model.pth')
                logger.info(f"  ✓ Mejor modelo guardado (val_loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"\nEarly stopping en epoch {epoch+1}")
                    break
        
        logger.info("\n" + "="*60)
        logger.info("ENTRENAMIENTO COMPLETADO")
        logger.info("="*60)
        logger.info(f"Mejor val_loss: {best_val_loss:.4f}")
        logger.info(f"Modelo guardado en: trading_model.pth")
        logger.info(f"Scaler guardado en: scaler.pkl")
        
        return model


if __name__ == "__main__":
    trainer = ModelTrainer(symbol='SOL/USDT', timeframe='5m')
    model = trainer.train_model(epochs=50, batch_size=32)
