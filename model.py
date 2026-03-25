import torch
import torch.nn as nn
import numpy as np
from loguru import logger

class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def get_model_path(symbol: str):
    return f"model_{symbol.replace('/', '_')}.pth"

def create_model():
    return LSTMModel()

def train_step(model, optimizer, df):
    if len(df) < 100:
        return 0.0
    data = df[['open', 'high', 'low', 'close', 'volume']].values
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    X = [data[i:i+60] for i in range(len(data) - 60)]
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(data[60:, 3:4], dtype=torch.float32)

    model.train()
    optimizer.zero_grad()
    pred = model(X)
    loss = nn.MSELoss()(pred, y)
    loss.backward()
    optimizer.step()
    logger.info(f"[{df['timestamp'].iloc[-1]}] Train loss: {loss.item():.6f}")
    return loss.item()

def predict(model, df):
    model.eval()
    data = df[['open', 'high', 'low', 'close', 'volume']].values
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    X = torch.tensor([data[-60:]], dtype=torch.float32)
    with torch.no_grad():
        pred = model(X)
    return float(pred[0][0])

def save_model(model, symbol: str):
    path = get_model_path(symbol)
    torch.save(model.state_dict(), path)
    logger.info(f"Модель сохранена: {path}")

def load_model(symbol: str):
    model = create_model()
    path = get_model_path(symbol)
    try:
        model.load_state_dict(torch.load(path, map_location='cpu'))
        logger.info(f"Модель загружена: {path}")
    except:
        logger.info(f"Новая модель для {symbol}")
    return model