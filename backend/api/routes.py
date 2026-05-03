from unittest import result

from fastapi import APIRouter, UploadFile, File
from backend.utils.file_utils import save_file
from backend.services.feature_services import analyze_features
from backend.services.report_service import generate_pdf_report

router = APIRouter()

@router.post("/analyze")
async def analyze(
    
    video: UploadFile = File(...),
   
):
   
    video_path = save_file(video)
    result = analyze_features(video_path)

    prediction = result["prediction"]
    report = result["report"]

    generate_pdf_report(
    prediction,
    report,
    filename="NeuroLens_Report.pdf"
)

    return result

   
from fastapi.responses import FileResponse

@router.get("/download-report")
def download_report():
    return FileResponse(
        "NeuroLens_Report.pdf",
        media_type="application/pdf",
        filename="NeuroLens_Report.pdf"
    )