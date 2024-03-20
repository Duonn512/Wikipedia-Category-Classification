import torch
from torch import nn as nn

class LMixed(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.bi_lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(2, batch_size, self.hidden_size, device=x.device)  # (2, batch_size, hidden_size)
        c0 = torch.zeros(2, batch_size, self.hidden_size, device=x.device)  # (2, batch_size, hidden_size)
        out, _ = self.bi_lstm(x, (h0, c0)) # (batch_size, T_x, hidden_size*2)
        # Max pooling over time
        out = torch.max(out, 1).values # (batch_size, hidden_size*2)
        out = self.fc(out) # (batch_size, num_classes)
        out = self.softmax(out) # (batch_size, num_classes)
        return out

