import streamlit as st
import requests

# -------------------------
# UI Setup
# -------------------------
st.set_page_config(page_title="MindScreen", layout="centered")

st.title("🧠 MindScreen")
st.subheader("Multimodal Mental Health Assessment")

# -------------------------
# Upload Video
# -------------------------
uploaded_video = st.file_uploader(
    "Upload Interview Video",
    type=["mp4", "mov", "avi"]
)

if uploaded_video:

    st.video(uploaded_video)

    # -------------------------
    # Analyze Button
    # -------------------------
    if st.button("Analyze"):

        files = {
            "video": (uploaded_video.name, uploaded_video, uploaded_video.type)
        }

        with st.spinner("Analyzing... This may take a moment ⏳"):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/analyze",
                    files=files
                )
            except Exception as e:
                st.error(f"Backend connection failed: {e}")
                st.stop()

        # -------------------------
        # Handle Response
        # -------------------------
        if response.status_code == 200:

            result = response.json()

            # Backend returns directly:
            # { "risk_score": ..., "label": ... }
            prediction = result

            risk_score = prediction.get("risk_score", "N/A")
            label = prediction.get("label", "N/A")

            st.success("✅ Assessment Complete")

            # -------------------------
            # Risk Output
            # -------------------------
            st.metric("Risk Level", label)
            st.metric("Risk Score", f"{risk_score}")

            if risk_score != "N/A":
                try:
                    st.progress(float(risk_score))
                except:
                    pass

            # -------------------------
            # Info Section
            # -------------------------
            st.subheader("Clinical Summary")
            st.write("Included in downloadable report.")

            st.subheader("Key Observations")
            st.write("Included in downloadable report.")

            st.subheader("Recommendations")
            st.write("Included in downloadable report.")

            # -------------------------
            # DOWNLOAD PDF BUTTON 🔥
            # -------------------------
            st.divider()

            st.subheader("Download Your Report")

            pdf_response = requests.get(
                "http://127.0.0.1:8000/download-report"
            )

            if pdf_response.status_code == 200:

                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_response.content,
                    file_name="NeuroLens_Report.pdf",
                    mime="application/pdf"
                )
            else:
                st.error("Failed to fetch PDF report.")

        else:
            st.error("❌ Analysis failed. Check backend.")

else:
    st.info("📤 Please upload a video file to begin.")