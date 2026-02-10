import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List
import pickle

class CryptoLSTM(nn.Module):
    """
    Modelo LSTM para predicción de precios de criptomonedas
    Incluye capas de atención y dropout para evitar overfitting
    """
    def __init__(self, input_size=10, hidden_size=128, num_layers=3, dropout=0.2):
        super(CryptoLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)  # 3 outputs: [buy_probability, hold_probability, sell_probability]
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Fully connected layers
        out = self.relu(self.fc1(context_vector))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        
        # Softmax para obtener probabilidades
        out = self.softmax(out)
        
        return out


class TradingModel:
    """
    Wrapper del modelo con funciones de entrenamiento y predicción
    """
    def __init__(self, input_size=10, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = CryptoLSTM(input_size=input_size).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = None
        
    def train_step(self, X_batch, y_batch):
        """Paso de entrenamiento"""
        self.model.train()
        
        X_batch = torch.FloatTensor(X_batch).to(self.device)
        y_batch = torch.LongTensor(y_batch).to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(X_batch)
        loss = self.criterion(outputs, y_batch)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def predict(self, X):
        """Predicción con el modelo"""
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).unsqueeze(0).to(self.device)
            prediction = self.model(X_tensor)
            
        return prediction.cpu().numpy()[0]
    
    def save_model(self, path='trading_model.pth'):
        """Guardar modelo"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        
    def load_model(self, path='trading_model.pth'):
        """Cargar modelo"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class FeatureEngineer:
    """
    Ingeniero de características para datos de trading
    """
    @staticmethod
    def calculate_rsi(prices, period=14):
        """Relative Strength Index"""
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.convolve(gain, np.ones(period)/period, mode='valid')
        avg_loss = np.convolve(loss, np.ones(period)/period, mode='valid')
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return np.concatenate([np.full(len(prices) - len(rsi), 50), rsi])
    
    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        """MACD indicator"""
        ema_fast = FeatureEngineer._ema(prices, fast)
        ema_slow = FeatureEngineer._ema(prices, slow)
        macd = ema_fast - ema_slow
        signal_line = FeatureEngineer._ema(macd, signal)
        
        return macd, signal_line
    
    @staticmethod
    def _ema(prices, period):
        """Exponential Moving Average"""
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        multiplier = 2 / (period + 1)
        
        for i in range(1, len(prices)):
            ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
            
        return ema
    
    @staticmethod
    def calculate_bollinger_bands(prices, period=20, std_dev=2):
        """Bandas de Bollinger"""
        sma = np.convolve(prices, np.ones(period)/period, mode='valid')
        std = np.array([np.std(prices[i:i+period]) for i in range(len(prices)-period+1)])
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        # Padding para mantener la longitud
        pad_size = len(prices) - len(sma)
        sma = np.concatenate([np.full(pad_size, sma[0]), sma])
        upper_band = np.concatenate([np.full(pad_size, upper_band[0]), upper_band])
        lower_band = np.concatenate([np.full(pad_size, lower_band[0]), lower_band])
        
        return sma, upper_band, lower_band
    
    @staticmethod
    def create_features(candles_data):
        """
        Crear todas las características para el modelo
        """
        closes = np.array([float(c['close']) for c in candles_data])
        volumes = np.array([float(c['volume']) for c in candles_data])
        highs = np.array([float(c['high']) for c in candles_data])
        lows = np.array([float(c['low']) for c in candles_data])
        
        # Indicadores técnicos
        rsi = FeatureEngineer.calculate_rsi(closes)
        macd, signal = FeatureEngineer.calculate_macd(closes)
        sma, upper_bb, lower_bb = FeatureEngineer.calculate_bollinger_bands(closes)
        
        # Volume features
        volume_ma = np.convolve(volumes, np.ones(20)/20, mode='valid')
        volume_ma = np.concatenate([np.full(len(volumes) - len(volume_ma), volume_ma[0]), volume_ma])
        
        # Price momentum
        momentum = np.concatenate([[0], np.diff(closes)])
        
        # Volatility
        volatility = np.array([np.std(closes[max(0, i-20):i+1]) for i in range(len(closes))])
        
        # Construir feature vector
        features = np.column_stack([
            closes,
            rsi,
            macd[-len(closes):],
            signal[-len(closes):],
            (closes - sma) / (sma + 1e-10),  # Normalized distance from SMA
            (closes - lower_bb) / (upper_bb - lower_bb + 1e-10),  # BB position
            volumes / (volume_ma + 1e-10),  # Volume ratio
            momentum,
            volatility,
            (highs - lows) / (closes + 1e-10)  # Range ratio
        ])
        
        return features
