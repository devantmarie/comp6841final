import torch
import torch.nn as nn
# definition of the models

# A very simple linear model to test

class SimpleLinear(nn.Module):
    def __init__(self, input_size=10, num_classes=13):
        super().__init__() 
        self.fc = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1) 
        return self.fc(x)
        