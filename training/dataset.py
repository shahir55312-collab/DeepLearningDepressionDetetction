import os
import pandas as pd
import torch
from torch.utils.data import Dataset

from backend.pipeline.preprocessing import (
    extract_audio_features,
    load_openface_features
)


class DAICDataset(Dataset):
    def __init__(self, data_dir, labels_file="Detailed_PHQ8_Labels.csv"):
        self.data_dir = data_dir

        # Load labels CSV (from training folder)
        labels_path = os.path.join("training", labels_file)
        self.labels = pd.read_csv(labels_path)

        # Convert PHQ score → binary label
        self.labels["label"] = self.labels["PHQ_8Total"].apply(
            lambda x: 1 if x >= 10 else 0
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]

        pid = row["Participant_ID"]
        label = row["label"]

        # -------------------------
        # Paths
        # -------------------------
        audio_path = os.path.join(
            self.data_dir, str(pid), f"{pid}_AUDIO.wav"
        )

        openface_csv_path = os.path.join(
            self.data_dir,
            str(pid),
            "features",
            f"{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv"
        )

        # -------------------------
        # Features
        # -------------------------
        
        if os.path.exists(audio_path):
            audio_feat = extract_audio_features(audio_path)
        else:
            audio_feat = torch.zeros((100, 40))  # fallback MFCC shape


        if os.path.exists(openface_csv_path):
            video_feat = load_openface_features(openface_csv_path)
        else:
            video_feat = torch.zeros((100, 512))  # fallback

        # Convert label to tensor
        audio_feat = (audio_feat- audio_feat.mean()) / (audio_feat.std() + 1e-6)
        video_feat = (video_feat - video_feat.mean()) / (video_feat.std() + 1e-6)
        label = torch.tensor(label, dtype=torch.long)

        return audio_feat, video_feat,label