import torch
import torch.nn as nn

class TextCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_sizes, output_size, embed_model, dropout_prob=0.5):
        super(TextCNN, self).__init__()
        
        vocab_size, embed_dim = embed_model.vectors.shape
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.embedding_layer.weight = nn.Parameter(torch.from_numpy(embed_model.vectors))  # all vectors
        
        self.embedding_layer.weight.requires_grad = False  # Freeze the embedding layer
        # cnn
        self.convs_1d_layers = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim), padding=(k - 2, 0))
            for k in kernel_sizes
        ])

        # fc
        self.fc_layer = nn.Linear(len(kernel_sizes) * num_filters, output_size)

        # 4. dropout and sigmoid layers
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.Sigmoid()

    def conv_and_pool(self, x, conv):
        """
        Convolutional + max pooling layer
        """
        # squeeze last dim to get size: (batch_size, num_filters, conv_seq_length)
        # conv_seq_length will be ~ 200
        x = torch.nn.ReLU()(conv(x)).squeeze(3)

        # 1D pool over conv_seq_length
        # squeeze to get size: (batch_size, num_filters)
        x_max = torch.nn.functional.max_pool1d(x, x.size(2)).squeeze(2)
        return x_max

    def forward(self, x):
        x = self.embedding_layer(x)  # (batch_size, seq_length, embedding_dim)
        conv_results = [self.conv_and_pool(x.unsqueeze(1), conv) for conv in self.convs_1d_layers]
        x = torch.cat(conv_results, 1)
        x = self.dropout(x)
        logit = self.fc_layer(x)
        out = self.activation(logit)
        return out