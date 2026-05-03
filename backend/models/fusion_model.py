import torch
import torch.nn as nn
class FusionModel(nn.Module):
    def __init__(self, audio_input_dim=40, video_input_dim=512, hidden_dim=128, num_classes=2,text_input_dim=768):
        super().__init__()
        self.audio_lstm = nn.LSTM(
            input_size=audio_input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.video_lstm = nn.LSTM(
            input_size=video_input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.text_lstm = nn.LSTM(
            input_size=text_input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim*4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, audio_x, video_x, text_x=None):
        audio_output, _ = self.audio_lstm(audio_x)
        video_output, _ = self.video_lstm(video_x)

        audio_pooled = audio_output.mean(dim=1)
        video_pooled = video_output.mean(dim=1)

    # handle text safely
        if text_x is not None:
            text_output, _ = self.text_lstm(text_x)
            text_pooled = text_output.mean(dim=1)
            fused = torch.cat((audio_pooled, video_pooled, text_pooled), dim=1)
        else:
            fused = torch.cat((audio_pooled, video_pooled), dim=1)

        out = self.classifier(fused)
        return out