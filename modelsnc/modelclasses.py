import torch
import torch.nn as nn
# definition of the models

# A very simple linear model 
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
    """
    A simple Multi-Layer Perceptron (MLP) model for the classification of non-coding RNAs.
    """    
    
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
    """
     A Convolutional Neural Network (CNN) model for the classification of non-coding RNAs.
     This model uses 3 convolutional layers to extract features from the input ncRNA sequences.
        
    """    
    def __init__(self, input_size=10, input_channels=4, num_classes=14,dropout_prob=0.5):
        super().__init__()
        self.num_classes = num_classes 
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=(7, 1), stride=1, padding=(3, 0))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(7, 1), stride=1, padding=(3, 0))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(7, 1), stride=1, padding=(3, 0))
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.fc1 = nn.Linear(256 * (input_size // 8), 512)  
        self.dropout = nn.Dropout(p=dropout_prob)
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
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def getModelName(self):
        return "Combine_CNN1"    

#********************************************
#
class RnnLSTM(nn.Module):
     """
      An LSTM-based model for the classification of non-coding RNAs.
      This model uses a Long Short-Term Memory (LSTM) network to capture sequential dependencies in ncRNA sequences. 
    """    
    
    def __init__(self, input_size=10, hidden_size=128, num_classes=14):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)  
        self.fc1 = nn.Linear(hidden_size, 256)  
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)  

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)  
        c0 = torch.zeros(1, x.size(0), self.hidden_size) 
        out, (hn, cn) = self.lstm(x, (h0, c0))  
        out = out[:, -1, :]  
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def getModelName(self):
        return "rnn_LSTM"

#********************************************
# 
class PositionalEncoding(nn.Module):
    """
     Auxiliary class to produce positional encodings used with transformer models.
     This class generates positional encodings using sine and cosine functions.
     The implementation is based on the COMP6841 Lab 7 course material.
    """    
    def __init__(self, embed_size, max_len=500):
       
        super().__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term  = 10000 ** (-torch.arange(0, embed_size, 2).float() / embed_size) 
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term) 
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)


#********************************************
# 
class Transformer(nn.Module):
    """
     A Transformer-based model for the classification of non-coding RNAs.
     The number of input channels is set to 4 to represent the four nucleotide bases.
    """    
    def __init__(self, input_size=10, input_channels=4, num_classes=14, d_model=64, nhead=4, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.embedding = nn.Linear(input_channels, d_model)
        self.positional_encoding = PositionalEncoding(embed_size=d_model, max_len=input_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(d_model * input_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)  
        x = self.positional_encoding(x)
        x = self.encoder(x)  
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def getModelName(self):
        return "Transformer"

#********************************************
#

