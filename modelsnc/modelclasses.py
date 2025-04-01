import torch
import torch.nn as nn
# definition of the models

# A very simple linear model to test
# Note: an extra class was added in case of unknown ncRNA class

class SimpleLinear(nn.Module):
    def __init__(self, input_size=10, num_classes=14):
        super().__init__() 
        self.fc = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1) 
        return self.fc(x)

#***********************************************
# A simple MLP

class SimpleMLP(nn.Module):
    def __init__(self, input_size=10, hidden_size=100, num_classes=14):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self,x):
        x = x.view(x.size(0), -1) 
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


