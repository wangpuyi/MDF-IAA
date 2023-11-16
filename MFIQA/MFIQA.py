import torch
import torch.nn as nn
import torch.nn.init as init

class MFIQAModule(nn.Module):
    def __init__(self):
        super(MFIQAModule, self).__init__()
        self.fc1 = nn.Linear(2848, 1024)  # First fully connected layer
        self.relu1 = nn.ReLU()            # First ReLU activation layer
        self.fc2 = nn.Linear(1024, 512)   # Second fully connected layer
        self.relu2 = nn.ReLU()            # Second ReLU activation layer
        self.fc3 = nn.Linear(512, 128)    # Third fully connected layer
        self.relu3 = nn.ReLU()            # Third ReLU activation layer
        self.fc4 = nn.Linear(128, 1) 
        # Initialize weights and biases
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaiming He initialization for ReLU activation
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)

        return x

