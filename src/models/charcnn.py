import torch
import torch.nn as nn

class CharCNN(nn.Module):
    def __init__(self, embed_dim, num_classes, seq_length,  dropout_prob=0.5):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=256, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3)
        )

        conv_out_size = (seq_length - 96) // 27 * 256

        self.classifier = nn.Sequential(
            nn.Linear(conv_out_size, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x) 
        x = self.conv2(x) 
        x = self.conv3(x) 
        x = self.conv4(x) 
        x = self.conv5(x) 
        x = self.conv6(x) 
        x = torch.flatten(x, start_dim=1) 
        x = self.classifier(x)
        return x