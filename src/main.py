"""
Document Extraction API - Main Application

This API reads Pub/Sub messages from GCS bucket, extracts metrics from PDF files,
and generates CSV files with the extracted data.
"""

import uvicorn
from src.controller.DocumentExtraction import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
