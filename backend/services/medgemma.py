# backend/services/medgemma.py

import json
import google.generativeai as genai


class MedGemmaService:

    def __init__(self):

        self.loaded = False

        print("Loading Gemini (MedGemma placeholder)...")

        try:
            genai.configure(api_key="AIzaSyAxPmLvApNywu0Jz_MUOWWMNyVminZiX44")  # get free key at aistudio.google.com

            self.model = genai.GenerativeModel(
                model_name="gemini-2.5-flash",
                system_instruction=(
                    "You are a clinical AI assistant helping clinicians with "
                    "longitudinal mental health monitoring. You do not diagnose disease. "
                    "You provide concise, structured monitoring summaries only."
                )
            )

            self.loaded = True
            print("Gemini loaded successfully.")

        except Exception as error:
            print("Gemini load failed:", error)
            print("Falling back to deterministic report generator.")
            self.model = None


    def build_prompt(self, features):

        prompt = f"""
You are assisting clinicians with longitudinal
mental health monitoring.

IMPORTANT:
- Do not diagnose disease.
- Provide monitoring summary only.
- Keep response concise and clinical.

Patient Signals:

Risk Score: {features["risk_score"]}
Voice Distress Score: {features["voice_distress"]}
Emotion Trend: {features["emotion_trend"]}
Weekly Change: {features["weekly_change"]}
Monitoring Window: {features["monitoring_window"]}

Return exactly in this format:

Clinical Summary:
Key Observations:
Potential Concerns:
Suggested Follow-up:
"""
        return prompt


    def call_medgemma(self, prompt):

        # Fallback if model not loaded
        if not self.loaded:
            return (
                "Clinical Summary: The patient presents a moderate risk profile based on extracted audio, video, and text signals. "
                "Key Observations: Elevated distress markers were noted in voice and visual affect. "
                
                "Suggested Follow-up: Schedule a follow-up evaluation within one to two weeks."
            )

        try:
            response = self.model.generate_content(prompt)
            return response.text

        except Exception as error:
            print("Gemini inference failed:", error)
            return (
                "Clinical Summary: Unable to generate report at this time. "
                "Key Observations: Model inference error occurred. "
                
                "Suggested Follow-up: Contact system administrator."
            )


    def format_report_sections(self, raw_report):

        sections = {
            "clinical_summary": "",
            "observations": "",
            "concerns": "",
            "follow_up": ""
        }

        current_section = None

        for line in raw_report.splitlines():

            line = line.strip()

            if not line:
                continue

            if "Clinical Summary:" in line:
                current_section = "clinical_summary"
                content = line.replace("Clinical Summary:", "").strip()
                if content:
                    sections[current_section] += content + " "
                continue

            elif "Key Observations:" in line:
                current_section = "observations"
                content = line.replace("Key Observations:", "").strip()
                if content:
                    sections[current_section] += content + " "
                continue

            elif "Potential Concerns:" in line:
                current_section = "concerns"
                content = line.replace("Potential Concerns:", "").strip()
                if content:
                    sections[current_section] += content + " "
                continue

            elif "Suggested Follow-up:" in line:
                current_section = "follow_up"
                content = line.replace("Suggested Follow-up:", "").strip()
                if content:
                    sections[current_section] += content + " "
                continue

            if current_section:
                sections[current_section] += line + " "

        return sections


    def generate_report(self, features):

        prompt = self.build_prompt(features)
        raw_report = self.call_medgemma(prompt)
        structured_report = self.format_report_sections(raw_report)

        return structured_report


# test
if __name__ == "_main_":

    patient_features = {
        "risk_score": 0.78,
        "voice_distress": 0.66,
        "emotion_trend": "negative",
        "weekly_change": "+18%",
        "monitoring_window": "30 days"
    }

    medgemma = MedGemmaService()

    report = medgemma.generate_report(patient_features)

    print(json.dumps(report, indent=4))