import torch

from backend.models.fusion_model import FusionModel
from backend.pipeline.preprocessing import (
    extract_audio_features,
    
    load_openface_features
)

from backend.services.report_service import generate_pdf_report
import subprocess
import os

def run_openface(video_path):
    output_dir = "temp_openface"
    os.makedirs(output_dir, exist_ok=True)

    command = [
        r"C:\Endeavour\OpenFace_2.2.0_win_x64\FeatureExtraction.exe",  # or full path
        "-f", os.path.abspath(video_path),
        "-out_dir", output_dir
    ]

    subprocess.run(command)

    base = os.path.basename(video_path).split(".")[0]
    csv_path = os.path.join(output_dir, base + ".csv")

    return csv_path

# -------------------------
# Load Trained Model

# -------------------------
def load_model():
    model = FusionModel()
    model.load_state_dict(
        torch.load("fusion_model.pth", map_location=torch.device("cpu"))
    )
    model.eval()
    return model


fusion_model = load_model()


# -------------------------
# Risk Label Function
# -------------------------
def get_risk_label(score):
    if score > 0.70:
        return "High Risk"
    elif score > 0.40:
        return "Moderate Risk"
    else:
        return "Low Risk"


# -------------------------
# Main Pipeline
# -------------------------
def run_pipeline(video_path):

    # -------------------------
    # Feature Extraction
    # -------------------------
    audio_features = extract_audio_features(video_path)

    # 🔥 OpenFace CSV path (IMPORTANT — adjust if needed)
    csv_path = run_openface(video_path)
    video_features = load_openface_features(csv_path)

    # Add batch dimension
    audio_input = audio_features.unsqueeze(0)
    video_input = video_features.unsqueeze(0)

    # -------------------------
    # Model Inference
    # -------------------------
    with torch.no_grad():
        output = fusion_model(
            audio_input,
            video_input,
            None   # ❗ no text
        )

        probabilities = torch.softmax(output, dim=1)
        risk_score = probabilities[0][1].item()

    # -------------------------
    # Result
    # -------------------------
    label = get_risk_label(risk_score)

    result = {
        "risk_score": round(risk_score, 4),
        "label": label
    }

    # -------------------------
    # Optional PDF Report
    # -------------------------
  

    return result

    # -------------------------
    # Feature Extraction
    # -------------------------
    