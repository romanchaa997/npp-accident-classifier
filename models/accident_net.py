# models/accident_net.py
import torch
import torch.nn as nn

class AccidentNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int,
                 num_layers: int, num_classes: int,
                 num_tags: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head_cls = nn.Linear(hidden_dim, num_classes)
        self.head_tags = nn.Linear(hidden_dim, num_tags)

    def forward(self, x):
        # x: (B, 50, 7)
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]             # (B, H)
        logits_cls = self.head_cls(last_hidden)    # (B, C_cls)
        logits_tags = self.head_tags(last_hidden)  # (B, C_tags)
        return logits_cls, logits_tags)
