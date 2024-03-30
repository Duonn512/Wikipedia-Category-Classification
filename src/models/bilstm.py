import torch
from torch import nn as nn

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, embed_model, num_layers, dropout_prob=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        vocab_size, embed_dim = embed_model.vectors.shape
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.embedding_layer.weight = nn.Parameter(torch.from_numpy(embed_model.vectors))  # all vectors
        
        self.embedding_layer.weight.requires_grad = False  # Freeze the embedding layer
        self.bi_lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        
        self.dropout = nn.Dropout(dropout_prob)  # Dropout layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.embedding_layer(x)  # (batch_size, seq_length, embedding_dim)
        h0 = torch.zeros(self.num_layers*2, batch_size, self.hidden_size, device=x.device)  # (num_layers*2, batch_size, hidden_size)
        c0 = torch.zeros(self.num_layers*2, batch_size, self.hidden_size, device=x.device)  # (num_layers*2, batch_size, hidden_size)
        out, _ = self.bi_lstm(x, (h0, c0)) # (batch_size, T_x, hidden_size*2)
        out = self.dropout(out)  # Apply dropout
        # Max pooling over time
        out = torch.max(out, 1).values # (batch_size, hidden_size*2)
        out = self.fc(out) # (batch_size, num_classes)
        out = self.softmax(out) # (batch_size, num_classes)
        return out
