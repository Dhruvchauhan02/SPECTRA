from fastapi import FastAPI

app = FastAPI(
    title="SPECTRA-AI API",
    description="AI-powered fake news and deepfake detection platform",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "SPECTRA-AI backend is running"}

@app.get("/health")
def health_check():
    return {"status": "OK"}
