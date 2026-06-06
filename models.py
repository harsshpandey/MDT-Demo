"""
Deep Learning Models for Wind Power Forecasting (V3 — NREL 5-MW)
===================================================================
High-capacity models targeting MAPE < 8%, R² > 0.96:
  1. LSTM     — hidden=256, layers=4, dropout=0.12
  2. GRU      — hidden=256, layers=4, dropout=0.12
  3. LSTMCNN  — LSTM(256,2) + Conv1D(128→64→32) + Attention Pool + FC
  4. GRUCNN   — GRU(256,2)  + Conv1D(128→64→32) + Attention Pool + FC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    """LSTM digital twin — moderate capacity tuned for ~7K training samples.

    Hidden=128, layers=2 gives ~150 K params. Previous hidden=256, layers=4
    overfit on the small dataset (val loss diverged 20× by epoch 10 while
    train loss kept falling). Bigger is not better when N_train ≈ 7 000.
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.15):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.bn(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class GRUModel(nn.Module):
    """GRU digital twin — moderate capacity (mirrors LSTM choice)."""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.15):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.bn(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class TemporalAttention(nn.Module):
    """Learned attention over time steps."""
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x: (batch, seq, hidden)
        scores = self.attn(x).squeeze(-1)          # (batch, seq)
        weights = F.softmax(scores, dim=-1)         # (batch, seq)
        ctx = torch.bmm(weights.unsqueeze(1), x)    # (batch, 1, hidden)
        return ctx.squeeze(1)                        # (batch, hidden)


class LSTMCNNModel(nn.Module):
    """LSTM + CNN + Temporal Attention hybrid (high capacity)."""
    def __init__(self, input_size, hidden_size=128, num_layers=2, window_size=24, dropout=0.15):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.conv1 = nn.Conv1d(hidden_size, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.attn = TemporalAttention(32)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        c = lstm_out.permute(0, 2, 1)
        c = self.relu(self.bn1(self.conv1(c)))
        c = self.relu(self.bn2(self.conv2(c)))
        c = self.relu(self.bn3(self.conv3(c)))
        c = c.permute(0, 2, 1)              # (batch, seq, 32)
        c = self.attn(c)                     # (batch, 32)
        c = self.dropout(c)
        c = self.relu(self.fc1(c))
        out = self.fc2(c)
        return out


class GRUCNNModel(nn.Module):
    """GRU + CNN + Temporal Attention hybrid (high capacity)."""
    def __init__(self, input_size, hidden_size=128, num_layers=2, window_size=24, dropout=0.15):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.conv1 = nn.Conv1d(hidden_size, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.attn = TemporalAttention(32)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        c = gru_out.permute(0, 2, 1)
        c = self.relu(self.bn1(self.conv1(c)))
        c = self.relu(self.bn2(self.conv2(c)))
        c = self.relu(self.bn3(self.conv3(c)))
        c = c.permute(0, 2, 1)
        c = self.attn(c)
        c = self.dropout(c)
        c = self.relu(self.fc1(c))
        out = self.fc2(c)
        return out


def get_model(model_name: str, input_size: int, window_size: int = 24):
    """Factory function to create models by name."""
    models = {
        'LSTM': lambda: LSTMModel(input_size),
        'GRU': lambda: GRUModel(input_size),
        'LSTMCNN': lambda: LSTMCNNModel(input_size, window_size=window_size),
        'GRUCNN': lambda: GRUCNNModel(input_size, window_size=window_size),
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    return models[model_name]()
