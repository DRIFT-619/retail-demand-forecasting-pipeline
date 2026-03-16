import torch
import torch.nn as nn

# Building LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        # The below line can also be written as super().__init__() as preferred in modern PyTorch 
        super(LSTMModel, self).__init__()
            
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
            
        self.fc = nn.Linear(hidden_size, 1)
            
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
            
        out, (h_n, c_n) = self.lstm(x)
            
        # Take last time step output
        last_out = out[:, -1, :]
            
        output = self.fc(last_out)
            
        return output.squeeze()
    