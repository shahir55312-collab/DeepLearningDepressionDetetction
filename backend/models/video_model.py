import torch.nn as nn
class VideoModel(nn.Module):
    def __init__(self,input_dim=512,hidden_dim=128,num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size = input_dim,
            hidden_size = hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim*2,64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self,x):
        output, _ = self.lstm(x)
        pooled = output[:,-1,:]
        out = self.classifier(pooled)
        return out