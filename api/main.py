from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
import uuid

from ai_models.deepfake_detection.pipeline import DeepfakePipeline

app = FastAPI(title="SPECTRA-AI API")

# Initialize pipeline once
pipeline = DeepfakePipeline(device="cpu")


@app.get("/")
def root():
    return {"message": "SPECTRA-AI is running"}


@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run deepfake analysis
        result = pipeline.analyze(temp_filename)

        # Remove temp file
        os.remove(temp_filename)

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
