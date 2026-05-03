
import os
import librosa
import numpy as np
import torch
import pandas as pd
from moviepy.editor import AudioFileClip
from transformers import BertTokenizer, BertModel


# -------------------------
# EXTENSIONS
# -------------------------
VIDEO_AUDIO_EXTENSIONS = {'.mp4', '.mov', '.mkv', '.avi', '.flv', '.webm'}
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}


# -------------------------
# AUDIO FEATURES (MFCC)
# -------------------------
def extract_audio_features(file_path, sr=16000, n_mfcc=40):
    ext = os.path.splitext(file_path)[1].lower()

    # If input is video → extract audio
    if ext in VIDEO_AUDIO_EXTENSIONS:
        try:
            clip = AudioFileClip(file_path)
            array = clip.to_soundarray(fps=sr)
            clip.close()

            if array.ndim > 1:
                y = array.mean(axis=1)
            else:
                y = array

        except Exception:
            y = np.zeros(sr * 5, dtype=np.float32)

    else:
        y, sr = librosa.load(file_path, sr=sr)

    # Fix length (5 seconds)
    y = librosa.util.fix_length(y, size=sr * 5)

    # MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T

    # Pad / trim to 100 timesteps
    if mfccs.shape[0] < 100:
        pad = np.zeros((100 - mfccs.shape[0], n_mfcc))
        mfccs = np.vstack((mfccs, pad))
    else:
        mfccs = mfccs[:100, :]

    return torch.tensor(mfccs, dtype=torch.float32)


# -------------------------
# VIDEO FEATURES (OpenFace CSV)
# -------------------------

def load_openface_features(csv_path):
    df = pd.read_csv(csv_path)

    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])

    features = df.values

    # Pad / trim to 100 timesteps
    if features.shape[0] < 100:
        pad = np.zeros((100 - features.shape[0], features.shape[1]))
        features = np.vstack((features, pad))
    else:
        features = features[:100, :]

    return torch.tensor(features, dtype=torch.float32)


# -------------------------
# TEXT FEATURES (BERT)
# -------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")
bert.eval()


def extract_text_features(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""

    if ext in {'.txt', '.md', '.csv'}:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            text = ""

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=100
    )

    with torch.no_grad():
        outputs = bert(**inputs)

    return outputs.last_hidden_state.squeeze(0)