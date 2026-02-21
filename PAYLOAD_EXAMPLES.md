# Sample Payloads for `/process` Endpoint

This document provides sample payloads for testing the Document Extraction API with fake GCS and fake Pub/Sub.

## Quick Start

1. **Set up fake GCS structure:**
   ```bash
   python setup_fake_gcs.py
   ```

2. **Place a PDF file in fake GCS:**
   ```bash
   mkdir -p ./fake_gcs_data/my-test-bucket/pdfs
   cp your-document.pdf ./fake_gcs_data/my-test-bucket/pdfs/
   ```

3. **Start the API:**
   ```bash
   python main.py
   ```

4. **Test with sample payloads (see below)**

## Payload Formats

### 1. Direct JSON Format (Simplest)

**Payload:**
```json
{
  "bucket": "my-test-bucket",
  "name": "pdfs/sample-document.pdf"
}
```

**cURL:**
```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{
    "bucket": "my-test-bucket",
    "name": "pdfs/sample-document.pdf"
  }'
```

**Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/process",
    json={
        "bucket": "my-test-bucket",
        "name": "pdfs/sample-document.pdf"
    }
)
print(response.json())
```

---

### 2. Pub/Sub Push Format (Base64 Encoded Data)

**Payload:**
```json
{
  "message": {
    "data": "eyJidWNrZXQiOiAibXktdGVzdC1idWNrZXQiLCAibmFtZSI6ICJwZGZzL3NhbXBsZS1kb2N1bWVudC5wZGYifQ==",
    "attributes": {},
    "messageId": "1234567890",
    "publishTime": "2025-02-21T10:00:00Z"
  },
  "subscription": "projects/my-project/subscriptions/my-subscription"
}
```

**Note:** The `data` field contains base64-encoded JSON: `{"bucket": "my-test-bucket", "name": "pdfs/sample-document.pdf"}`

**cURL:**
```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{
    "message": {
      "data": "eyJidWNrZXQiOiAibXktdGVzdC1idWNrZXQiLCAibmFtZSI6ICJwZGZzL3NhbXBsZS1kb2N1bWVudC5wZGYifQ=="
    }
  }'
```

**Python (with base64 encoding):**
```python
import json
import base64
import requests

message = {
    "bucket": "my-test-bucket",
    "name": "pdfs/sample-document.pdf"
}
data_str = json.dumps(message)
data_b64 = base64.b64encode(data_str.encode('utf-8')).decode('utf-8')

response = requests.post(
    "http://localhost:8000/process",
    json={
        "message": {
            "data": data_b64
        }
    }
)
print(response.json())
```

---

### 3. Pub/Sub Attributes Format

**Payload:**
```json
{
  "message": {
    "attributes": {
      "bucket": "my-test-bucket",
      "name": "pdfs/sample-document.pdf"
    },
    "messageId": "1234567890",
    "publishTime": "2025-02-21T10:00:00Z"
  }
}
```

**cURL:**
```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{
    "message": {
      "attributes": {
        "bucket": "my-test-bucket",
        "name": "pdfs/sample-document.pdf"
      }
    }
  }'
```

---

### 4. Direct Base64 Data Format

**Payload:**
```json
{
  "data": "eyJidWNrZXQiOiAibXktdGVzdC1idWNrZXQiLCAibmFtZSI6ICJwZGZzL3NhbXBsZS1kb2N1bWVudC5wZGYifQ=="
}
```

**cURL:**
```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{
    "data": "eyJidWNrZXQiOiAibXktdGVzdC1idWNrZXQiLCAibmFtZSI6ICJwZGZzL3NhbXBsZS1kb2N1bWVudC5wZGYifQ=="
  }'
```

---

### 5. Full Pub/Sub Format (With Event Attributes)

**Payload:**
```json
{
  "message": {
    "data": "eyJidWNrZXQiOiAibXktdGVzdC1idWNrZXQiLCAibmFtZSI6ICJwZGZzL3NhbXBsZS1kb2N1bWVudC5wZGYifQ==",
    "attributes": {
      "eventType": "OBJECT_FINALIZE",
      "eventId": "1234567890",
      "bucketId": "my-test-bucket",
      "objectId": "pdfs/sample-document.pdf"
    },
    "messageId": "1234567890",
    "publishTime": "2025-02-21T10:00:00Z"
  },
  "subscription": "projects/my-project/subscriptions/my-subscription"
}
```

---

## Helper: Base64 Encoding

To create base64-encoded data for Pub/Sub format:

**Python:**
```python
import json
import base64

message = {
    "bucket": "my-test-bucket",
    "name": "pdfs/sample-document.pdf"
}
data_str = json.dumps(message)
data_b64 = base64.b64encode(data_str.encode('utf-8')).decode('utf-8')
print(data_b64)
# Output: eyJidWNrZXQiOiAibXktdGVzdC1idWNrZXQiLCAibmFtZSI6ICJwZGZzL3NhbXBsZS1kb2N1bWVudC5wZGYifQ==
```

**Bash:**
```bash
echo '{"bucket": "my-test-bucket", "name": "pdfs/sample-document.pdf"}' | base64
```

---

## Expected Response

All payloads should return a response like:

```json
{
  "status": "success",
  "input_file": "gs://my-test-bucket/pdfs/sample-document.pdf",
  "output_file": "gs://my-test-bucket/output/sample-document_metrics.csv",
  "metrics_extracted": 5,
  "metrics": [
    {
      "metric_name": "revenue",
      "value": "$1,500,000",
      "attributes": {
        "metric_name": "revenue",
        "value": "$1,500,000",
        "definition": "Total amount of money generated...",
        "synonyms": "revenue, total revenue, sales"
      }
    }
  ]
}
```

---

## Fake GCS File Structure

When using fake GCS (`USE_FAKE_GCS=true`), files are stored locally:

```
./fake_gcs_data/
  my-test-bucket/
    pdfs/
      sample-document.pdf          # Input PDF files
    output/
      sample-document_metrics.csv  # Generated CSV files
```

**To add a PDF:**
```bash
mkdir -p ./fake_gcs_data/my-test-bucket/pdfs
cp your-file.pdf ./fake_gcs_data/my-test-bucket/pdfs/
```

**Output location:**
```bash
cat ./fake_gcs_data/my-test-bucket/output/sample-document_metrics.csv
```

---

## Testing Script

Use the provided test script:

```bash
./test_payloads.sh
```

Or manually test each format using the examples above.
