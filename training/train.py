import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from backend.models.fusion_model import FusionModel
from training.dataset import DAICDataset


def main():
    fusion_model = FusionModel()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0]))  # handle class imbalance
    optimizer = optim.Adam(fusion_model.parameters(), lr=0.0001)

    dataset = DAICDataset(data_dir="data/daic")
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    fusion_model.train()
    for epoch in range(10):
        total_loss = 0.0
        for audio, video,label in loader:
            optimizer.zero_grad()
            out = fusion_model(audio, video)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = torch.argmax(out, dim=1)
           
            print("pred : ",preds.tolist(), "label : ", label.tolist())
        print(f"Epoch {epoch + 1}/10 - Loss: {total_loss:.4f}")

    torch.save(fusion_model.state_dict(), "fusion_model.pth")


if __name__ == '__main__':
    main()