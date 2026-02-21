"""
Setup script for fake GCS - creates directory structure and sample PDF

This script helps set up the fake GCS filesystem structure for testing.
"""

import os
import shutil
from pathlib import Path

FAKE_GCS_ROOT = os.getenv("FAKE_GCS_ROOT", "./fake_gcs_data")
BUCKET_NAME = "my-test-bucket"
SAMPLE_PDF_PATH = "pdfs/sample-document.pdf"


def create_fake_gcs_structure():
    """Create fake GCS directory structure"""
    fake_gcs_path = Path(FAKE_GCS_ROOT) / BUCKET_NAME / Path(SAMPLE_PDF_PATH).parent
    fake_gcs_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Created fake GCS directory structure: {fake_gcs_path}")
    print(f"\nTo add a PDF file, place it at:")
    print(f"  {Path(FAKE_GCS_ROOT) / BUCKET_NAME / SAMPLE_PDF_PATH}")
    print(f"\nOutput CSV files will be written to:")
    print(f"  {Path(FAKE_GCS_ROOT) / BUCKET_NAME / 'output'}/")
    
    # Create a sample text file explaining the structure
    readme_path = Path(FAKE_GCS_ROOT) / "README.txt"
    readme_path.write_text(f"""Fake GCS Storage Structure

This directory mimics Google Cloud Storage for local testing.

Structure:
  {FAKE_GCS_ROOT}/
    {BUCKET_NAME}/
      pdfs/
        your-document.pdf          # Place your PDF files here
      output/
        your-document_metrics.csv  # Generated CSV files appear here

To use:
1. Place your PDF files in: {FAKE_GCS_ROOT}/{BUCKET_NAME}/pdfs/
2. Send a request to the API with:
   - bucket: "{BUCKET_NAME}"
   - name: "pdfs/your-document.pdf"
3. Check the output folder for generated CSV files

Environment Variables:
  USE_FAKE_GCS=true (default)
  FAKE_GCS_ROOT=./fake_gcs_data (default)
""")
    
    print(f"\nCreated README at: {readme_path}")


def create_sample_pdf():
    """Create a sample PDF file for testing"""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        sample_pdf_path = Path(FAKE_GCS_ROOT) / BUCKET_NAME / SAMPLE_PDF_PATH
        sample_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a simple PDF with sample text
        c = canvas.Canvas(str(sample_pdf_path), pagesize=letter)
        width, height = letter
        
        # Add some sample text with metrics
        text = c.beginText(50, height - 50)
        text.setFont("Helvetica", 12)
        
        sample_content = [
            "Sample Business Report",
            "",
            "Financial Metrics:",
            "Revenue: $1,500,000",
            "Profit: $250,000",
            "Expenses: $1,250,000",
            "",
            "Operational Metrics:",
            "Employee Count: 150",
            "Customer Count: 5,000",
            "Growth Rate: 15.5%",
            "",
            "Performance Metrics:",
            "Customer Satisfaction: 87%",
            "Operating Margin: 18.5%",
            "Market Share: 12.3%"
        ]
        
        for line in sample_content:
            text.textLine(line)
        
        c.drawText(text)
        c.save()
        
        print(f"\nCreated sample PDF at: {sample_pdf_path}")
        print("You can now test the API with this file!")
        
    except ImportError:
        print("\nNote: reportlab not installed. Skipping sample PDF creation.")
        print("Install it with: pip install reportlab")
        print(f"\nPlace your own PDF file at: {Path(FAKE_GCS_ROOT) / BUCKET_NAME / SAMPLE_PDF_PATH}")


if __name__ == "__main__":
    print("Setting up fake GCS structure...")
    create_fake_gcs_structure()
    create_sample_pdf()
    print("\nSetup complete!")
