#!/bin/bash

# Test script for the Document Extraction API
# Tests various payload formats for the /process endpoint

API_URL="http://localhost:8000"

echo "=========================================="
echo "Testing Document Extraction API"
echo "=========================================="
echo ""

# Test 1: Direct JSON format
echo "Test 1: Direct JSON format"
echo "---------------------------"
curl -X POST "${API_URL}/process" \
  -H "Content-Type: application/json" \
  -d '{
    "bucket": "my-test-bucket",
    "name": "pdfs/sample-document.pdf"
  }' | jq .
echo ""
echo ""

# Test 2: Pub/Sub push format with base64 data
echo "Test 2: Pub/Sub push format (base64 data)"
echo "-------------------------------------------"
# Base64 encoded: {"bucket": "my-test-bucket", "name": "pdfs/sample-document.pdf"}
curl -X POST "${API_URL}/process" \
  -H "Content-Type: application/json" \
  -d '{
    "message": {
      "data": "eyJidWNrZXQiOiAibXktdGVzdC1idWNrZXQiLCAibmFtZSI6ICJwZGZzL3NhbXBsZS1kb2N1bWVudC5wZGYifQ=="
    }
  }' | jq .
echo ""
echo ""

# Test 3: Pub/Sub attributes format
echo "Test 3: Pub/Sub attributes format"
echo "----------------------------------"
curl -X POST "${API_URL}/process" \
  -H "Content-Type: application/json" \
  -d '{
    "message": {
      "attributes": {
        "bucket": "my-test-bucket",
        "name": "pdfs/sample-document.pdf"
      }
    }
  }' | jq .
echo ""
echo ""

# Test 4: Health check
echo "Test 4: Health check"
echo "---------------------"
curl -X GET "${API_URL}/health" | jq .
echo ""
echo ""

echo "=========================================="
echo "Tests complete!"
echo "=========================================="
