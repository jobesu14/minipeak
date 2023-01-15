import torch
import torch.nn as nn

class CNN(nn.Module):
    """ 1D CNN model for minipeak detection. """
    
    def __init__(self, window_size: int):
        super(CNN, self).__init__()
        self.window_size = window_size
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.fc1 = nn.Linear(64 * int(window_size / 2), 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)

    def forward(self, x):
        # Convolution layers
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return torch.sigmoid(x)
