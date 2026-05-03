import torch
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

UPLOAD_DIR = os.path.join(BASE_DIR, "../uploads")
REPORT_DIR = os.path.join(BASE_DIR, "../reports")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


AUDIO_INPUT_DIM = 40
VIDEO_INPUT_DIM = 512
TEXT_INPUT_DIM = 768
HIDDEN_DIM = 128
NUM_CLASSES = 2

# Inference threshold
RISK_THRESHOLD = 0.5