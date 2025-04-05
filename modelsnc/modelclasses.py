import torch
import torch.nn as nn
# definition of the models

# A very simple linear model to test
# Note: an extra class was added in case of unknown ncRNA class

class SimpleLinear(nn.Module):
    def __init__(self, input_size=10, num_classes=14):
        super().__init__() 
        self.num_classes = num_classes 
        self.fc = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1) 
        return self.fc(x)

    def getModelName(self):
        return "Simple_Linear_Model"
        

#***********************************************
# A simple MLP

class SimpleMLP(nn.Module):
    def __init__(self, input_size=10, hidden_size=100, num_classes=14):
        super().__init__()
        self.num_classes = num_classes 
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self,x):
        x = x.view(x.size(0), -1) 
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def getModelName(self):
        return "Simple_MLP"    

#*********************************************
# 
class CombineCNN1(nn.Module):
    def __init__(self, input_size=10, input_channels=4, num_classes=14):
        super().__init__()
        self.num_classes = num_classes 
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=(7, 1), stride=1, padding=(3, 0))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(7, 1), stride=1, padding=(3, 0))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(7, 1), stride=1, padding=(3, 0))
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.fc1 = nn.Linear(256 * (input_size // 8), 512)  
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(3) 
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1) 
        self.fc1 = nn.Linear(x.size(1), 512)  
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def getModelName(self):
        return "Combine_CNN1"    

#********************************************
#
class rnnLSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=100, num_classes=14):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)  
        self.fc = nn.Linear(hidden_size, num_classes)  

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)  
        c0 = torch.zeros(1, x.size(0), self.hidden_size) 
        out, (hn, cn) = self.lstm(x, (h0, c0))  
        out = out[:, -1, :]  
        out = self.fc(out)  
        return out

    def getModelName(self):
        return "rnn_LSTM"

#********************************************


