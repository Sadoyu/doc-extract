"""
Example usage script for testing the Document Extraction API locally.

This script demonstrates how to call the API endpoints.
"""

import json
import base64
import requests

# API endpoint
API_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")

def test_process_direct(bucket: str, file_path: str):
    """Test direct processing endpoint"""
    print(f"Testing direct processing for gs://{bucket}/{file_path}...")
    response = requests.post(
        f"{API_URL}/process-direct",
        params={"bucket": bucket, "file_path": file_path}
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Success! Extracted {result['metrics_extracted']} metrics")
        print(f"Output file: {result['output_file']}")
        print(f"Metrics: {json.dumps(result['metrics'], indent=2)}")
    else:
        print(f"Error: {response.text}")
    print()

def test_process_pubsub_message(bucket: str, file_path: str):
    """Test Pub/Sub message processing endpoint"""
    print(f"Testing Pub/Sub message processing for gs://{bucket}/{file_path}...")
    
    # Create message data
    message_data = {
        "bucket": bucket,
        "name": file_path
    }
    
    # Encode as base64 (simulating Pub/Sub format)
    data_str = json.dumps(message_data)
    data_b64 = base64.b64encode(data_str.encode('utf-8')).decode('utf-8')
    
    # Pub/Sub push format
    pubsub_message = {
        "message": {
            "data": data_b64,
            "attributes": {}
        }
    }
    
    response = requests.post(
        f"{API_URL}/process",
        json=pubsub_message,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Success! Extracted {result['metrics_extracted']} metrics")
        print(f"Output file: {result['output_file']}")
    else:
        print(f"Error: {response.text}")
    print()

def test_process_direct_json(bucket: str, file_path: str):
    """Test direct JSON format"""
    print(f"Testing direct JSON format for gs://{bucket}/{file_path}...")
    
    message_data = {
        "bucket": bucket,
        "name": file_path
    }
    
    response = requests.post(
        f"{API_URL}/process",
        json=message_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Success! Extracted {result['metrics_extracted']} metrics")
        print(f"Output file: {result['output_file']}")
    else:
        print(f"Error: {response.text}")
    print()

if __name__ == "__main__":
    # Example usage - replace with your actual bucket and file path
    BUCKET = "data"
    FILE_PATH = "pdfs/example-document.pdf"
    
    print("=" * 60)
    print("Document Extraction API - Example Usage")
    print("=" * 60)
    print()
    
    # Test health check
    test_health_check()
    
    # Uncomment the following lines to test actual processing:
    test_process_direct(BUCKET, FILE_PATH)
    test_process_pubsub_message(BUCKET, FILE_PATH)
    test_process_direct_json(BUCKET, FILE_PATH)
    
    print("Note: Update BUCKET and FILE_PATH variables with your actual values")
    print("Make sure the API server is running: python main.py")
