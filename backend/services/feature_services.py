from backend.pipeline.inference import run_pipeline
from backend.services.medgemma import MedGemmaService


def analyze_features(video_path):

    # 1️⃣ Run pipeline (your model)
    pipeline_result = run_pipeline(video_path)

    print("PIPELINE RESULT:", pipeline_result)  # debug

    # 2️⃣ Extract prediction safely
    # (your pipeline returns risk_score + label)
    risk_score = pipeline_result.get("risk_score", 0.0)
    label = pipeline_result.get("label", "Unknown")

    prediction = {
        "risk_score": risk_score,
        "label": label
    }

    # 3️⃣ Build features for Gemini
    features = {
        "risk_score": risk_score,
        "voice_distress": 0.5,         # replace later with real audio
        "emotion_trend": "neutral",    # replace later
        "weekly_change": "+5%",        # replace later
        "monitoring_window": "7 days"
    }

    # 4️⃣ Call Gemini
    medgemma = MedGemmaService()
    report = medgemma.generate_report(features)

    print("GEMINI REPORT:", report)  # debug

    # 5️⃣ Return structured result
    return {
        "prediction": prediction,
        "report": report
    }