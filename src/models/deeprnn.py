import torch
from torch import nn as nn

class DeepRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers, embed_model, dropout_prob=0.5):
        super(DeepRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        vocab_size, embed_dim = embed_model.vectors.shape
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.embedding_layer.weight = nn.Parameter(torch.from_numpy(embed_model.vectors))  # all vectors
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.embedding_layer(x)  # (batch_size, seq_length, embedding_dim)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        out, _ = self.rnn(x, h0)
        if self.training:
            out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out
