import json, logging
from fastapi import FastAPI, HTTPException, Request
from src.config.Config import Config
from src.service.DocumentService import DocumentService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Document Extraction API")

@app.post("/process")
async def process_document(request: Request):
    """Process a document from Pub/Sub message"""
    try:
        body_bytes = await request.body()
        if not body_bytes or not body_bytes.strip():
            raise HTTPException(
                status_code=400,
                detail="Request body is empty. Send JSON with 'bucket' and 'name', or a Pub/Sub message with 'message' or 'data'."
            )
        try:
            body = json.loads(body_bytes)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Request body must be valid JSON. Error: {e}"
            )

        # Handle Pub/Sub message format
        if 'message' in body:
            # Pub/Sub push format
            message = body['message']
            if 'data' in message:
                import base64
                data = base64.b64decode(message['data']).decode('utf-8')
                message_data = json.loads(data)
            else:
                message_data = message.get('attributes', {})
        elif 'data' in body:
            # Direct data format
            import base64
            data = base64.b64decode(body['data']).decode('utf-8')
            message_data = json.loads(data)
        else:
            # Assume direct JSON format
            message_data = body

        result = DocumentService.process_pubsub_message(message_data)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /process endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-direct")
async def process_direct(bucket: str, file_path: str):
    """Process a document directly by specifying bucket and file path as query parameters"""
    try:
        message_data = {
            "bucket": bucket,
            "name": file_path
        }
        result = DocumentService.process_pubsub_message(message_data)
        return result
    except Exception as e:
        logger.error(f"Error in /process-direct endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ollama_model": Config.OLLAMA_MODEL,
        "ollama_timeout_seconds": Config.OLLAMA_TIMEOUT,
        "extract_max_chars": Config.EXTRACT_MAX_CHARS or None,
        "use_fake_gcs": Config.USE_FAKE_GCS,
        "use_fake_pubsub": Config.USE_FAKE_PUBSUB,
        "fake_gcs_root": Config.FAKE_GCS_ROOT if Config.USE_FAKE_GCS else None
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Document Extraction API",
        "endpoints": {
            "/process": "POST - Process document from Pub/Sub message",
            "/process-direct": "POST - Process document by bucket and file path",
            "/health": "GET - Health check"
        }
    }
