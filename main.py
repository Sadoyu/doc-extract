"""
Document Extraction API - Main Application

This API reads Pub/Sub messages from GCS bucket, extracts metrics from PDF files,
and generates CSV files with the extracted data.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import yaml
from fastapi import FastAPI, HTTPException, Request
from google.cloud import storage
import PyPDF2
from pydantic import BaseModel
import langextract as lx
from langextract.core.data import ExampleData, Extraction, EXTRACTIONS_KEY, FormatType
import LenientOllamaFormatHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Document Extraction API")

# Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "1200"))  # per-request timeout (seconds)
METRICS_YAML_PATH = os.getenv("METRICS_YAML_PATH", "metrics.yaml")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "data")
OUTPUT_FOLDER = "output"
# Limit text length for extraction to avoid timeouts on large docs (0 = no limit)
EXTRACT_MAX_CHARS = int(os.getenv("EXTRACT_MAX_CHARS", "1000"))

# Fake GCS/PubSub Configuration
USE_FAKE_GCS = os.getenv("USE_FAKE_GCS", "true").lower() == "true"
FAKE_GCS_ROOT = os.getenv("FAKE_GCS_ROOT", "/Users/saikiranchandolu/Workspace/data-bucket")
USE_FAKE_PUBSUB = os.getenv("USE_FAKE_PUBSUB", "true").lower() == "true"


class PubSubMessageData(BaseModel):
    """Pub/Sub message data structure"""
    bucket: str
    name: str  # File path in bucket


def load_metrics_config(yaml_path: str) -> Dict[str, Any]:
    """Load metrics configuration from YAML file"""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def create_examples_from_metrics(metrics_config: Dict[str, Any]) -> List[ExampleData]:
    """Create few-shot examples from metrics configuration"""
    examples = []
    metrics = metrics_config.get('metrics', [])
    
    # Create example for each metric type
    for metric in metrics[:3]:  # Use first 3 metrics as examples
        metric_name = metric['name']
        synonyms = metric.get('synonyms', [])
        definition = metric.get('definition', '')
        example_value = metric.get('example') or ''
        if not isinstance(example_value, (str, int, float)):
            example_value = str(example_value) if example_value else ''
        
        # Create example text
        example_text = f"The {metric_name} for this period is {example_value}. "
        example_text += f"This represents the {definition.lower()}."
        
        # Create extraction
        extraction = Extraction(
            extraction_class=metric_name,
            extraction_text=example_value,
            attributes={
                "metric_name": metric_name,
                "value": example_value,
                "definition": definition,
                "synonyms": ", ".join(synonyms)
            }
        )
        
        examples.append(ExampleData(
            text=example_text,
            extractions=[extraction]
        ))
    
    return examples


def build_prompt_description(metrics_config: Dict[str, Any]) -> str:
    """Build prompt description for metric extraction"""
    metrics = metrics_config.get('metrics', [])
    
    prompt = "Extract the following business metrics from the document:\n\n"
    
    for metric in metrics:
        metric_name = metric['name']
        synonyms = ", ".join(metric.get('synonyms', []))
        definition = metric.get('definition', '')
        
        prompt += f"- {metric_name.upper()}: {definition}\n"
        prompt += f"  Synonyms: {synonyms}\n"
        prompt += f"  Example: {metric.get('example', 'N/A')}\n\n"
    
    prompt += "\nFor each metric found, extract:\n"
    prompt += "- The metric name (use the canonical name from the list above)\n"
    prompt += "- The value (as it appears in the document)\n"
    prompt += "- The context or description if available\n"
    prompt += "\nIMPORTANT: Always output a string for every metric value. Never use null. "
    prompt += "If a value is not found in the document, use empty string \"\" or omit that metric from the output.\n"
    
    return prompt


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text content from PDF bytes"""
    text_content = []
    
    try:
        pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        pdf_file.write(pdf_bytes)
        pdf_file.close()
        
        with open(pdf_file.name, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    text_content.append(f"--- Page {page_num + 1} ---\n{text}\n")
        
        os.unlink(pdf_file.name)
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise
    
    return "\n".join(text_content)


def download_pdf_from_gcs(bucket_name: str, file_path: str) -> bytes:
    """Download PDF file from GCS bucket (real or fake)"""
    try:
        if USE_FAKE_GCS:
            # Use local filesystem as fake GCS
            fake_gcs_path = Path(FAKE_GCS_ROOT) / bucket_name / file_path
            fake_gcs_path.parent.mkdir(parents=True, exist_ok=True)
            
            if not fake_gcs_path.exists():
                raise FileNotFoundError(f"File not found in fake GCS: {fake_gcs_path}")
            
            logger.info(f"Downloading {file_path} from fake GCS bucket {bucket_name}")
            pdf_bytes = fake_gcs_path.read_bytes()
            logger.info(f"Successfully downloaded {len(pdf_bytes)} bytes from fake GCS")
            
            return pdf_bytes
        else:
            # Use real GCS
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(file_path)
            
            logger.info(f"Downloading {file_path} from bucket {bucket_name}")
            pdf_bytes = blob.download_as_bytes()
            logger.info(f"Successfully downloaded {len(pdf_bytes)} bytes")
            
            return pdf_bytes
    except Exception as e:
        logger.error(f"Error downloading PDF from GCS: {e}")
        raise


def _sanitize_extraction_data_for_resolver(
    extraction_data: Any,
    index_suffix: str,
    attributes_suffix: str,
) -> Any:
    """Coerce None extraction values to empty string so LangExtract resolver does not raise."""
    if not extraction_data:
        return extraction_data
    sanitized = []
    for group in extraction_data:
        if not isinstance(group, dict):
            sanitized.append(group)
            continue
        new_group = {}
        for k, v in group.items():
            if index_suffix and k.endswith(index_suffix):
                new_group[k] = v
                continue
            if attributes_suffix and k.endswith(attributes_suffix):
                new_group[k] = v
                continue
            if v is None:
                new_group[k] = ""
            elif isinstance(v, (str, int, float)):
                new_group[k] = v
            else:
                new_group[k] = v
        sanitized.append(new_group)
    return sanitized


def extract_metrics_from_text(text: str, metrics_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract metrics from text using langextract with Ollama"""
    try:
        # Optionally truncate to avoid timeouts on very long documents
        if EXTRACT_MAX_CHARS and len(text) > EXTRACT_MAX_CHARS:
            text = text[:EXTRACT_MAX_CHARS]
            logger.info(f"Truncated text to {EXTRACT_MAX_CHARS} chars for extraction")

        # Create examples and prompt
        examples = create_examples_from_metrics(metrics_config)
        prompt_description = build_prompt_description(metrics_config)
        
        logger.info(f"Extracting metrics using Ollama model: {OLLAMA_MODEL} (timeout={OLLAMA_TIMEOUT}s)")

        # Patch Resolver to sanitize None values (LLM sometimes returns null)
        from langextract import resolver as resolver_module
        original_extract = resolver_module.Resolver.extract_ordered_extractions

        def patched_extract_ordered_extractions(
            self: Any,
            extraction_data: Any,
        ) -> Any:
            index_suffix = getattr(self, "extraction_index_suffix", "") or ""
            attributes_suffix = getattr(
                self.format_handler, "attribute_suffix", "_attributes"
            ) or "_attributes"
            sanitized = _sanitize_extraction_data_for_resolver(
                extraction_data, index_suffix, attributes_suffix
            )
            return original_extract(self, sanitized)

        resolver_module.Resolver.extract_ordered_extractions = patched_extract_ordered_extractions
        try:
            result = lx.extract(
                text_or_documents=text,
                model_id=OLLAMA_MODEL,
                prompt_description=prompt_description,
                examples=examples,
                max_char_buffer=4000,  # Larger chunks = fewer Ollama calls, less timeout risk
                extraction_passes=1,  # Single pass to reduce total time
                max_workers=2,  # Fewer parallel calls to avoid overloading local Ollama
                language_model_params={
                    "timeout": OLLAMA_TIMEOUT,
                    "model_url": OLLAMA_URL
                },
                resolver_params={
                    "format_handler": LenientOllamaFormatHandler
                }
            )

            # Convert extractions to list of dictionaries
            extracted_metrics = []
            if isinstance(result, list):
                for doc in result:
                    if hasattr(doc, 'extractions') and doc.extractions:
                        for extraction in doc.extractions:
                            val = extraction.extraction_text
                            if val is None:
                                val = ""
                            elif not isinstance(val, (str, int, float)):
                                val = str(val)
                            metric_data = {
                                "metric_name": extraction.extraction_class,
                                "value": val,
                                "attributes": extraction.attributes or {}
                            }
                            extracted_metrics.append(metric_data)
            elif hasattr(result, 'extractions') and result.extractions:
                for extraction in result.extractions:
                    val = extraction.extraction_text
                    if val is None:
                        val = ""
                    elif not isinstance(val, (str, int, float)):
                        val = str(val)
                    metric_data = {
                        "metric_name": extraction.extraction_class,
                        "value": val,
                        "attributes": extraction.attributes or {}
                    }
                    extracted_metrics.append(metric_data)

            logger.info(f"Extracted {len(extracted_metrics)} metrics")
            return extracted_metrics
        finally:
            resolver_module.Resolver.extract_ordered_extractions = original_extract

    except Exception as e:
        logger.error(f"Error extracting metrics: {e}")
        raise


def generate_csv(metrics: List[Dict[str, Any]], output_path: str):
    """Generate CSV file from extracted metrics"""
    import csv
    
    if not metrics:
        logger.warning("No metrics to write to CSV")
        return
    
    # Get all unique attribute keys
    all_keys = set()
    for metric in metrics:
        all_keys.add("metric_name")
        all_keys.add("value")
        if metric.get("attributes"):
            all_keys.update(metric["attributes"].keys())
    
    fieldnames = ["metric_name", "value"] + sorted([k for k in all_keys if k not in ["metric_name", "value"]])
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for metric in metrics:
            row = {
                "metric_name": metric.get("metric_name", ""),
                "value": metric.get("value", "")
            }
            if metric.get("attributes"):
                row.update(metric["attributes"])
            writer.writerow(row)
    
    logger.info(f"Generated CSV file: {output_path}")


def upload_csv_to_gcs(bucket_name: str, csv_path: str, original_file_path: str):
    """Upload CSV file to GCS bucket in output folder (real or fake)"""
    try:
        # Create output path: output/filename.csv
        original_filename = Path(original_file_path).stem
        output_blob_name = f"{OUTPUT_FOLDER}/{original_filename}_metrics.csv"
        
        if USE_FAKE_GCS:
            # Use local filesystem as fake GCS
            fake_gcs_output_path = Path(FAKE_GCS_ROOT) / bucket_name / output_blob_name
            fake_gcs_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy CSV file to fake GCS location
            import shutil
            shutil.copy2(csv_path, fake_gcs_output_path)
            
            logger.info(f"Uploaded CSV to fake GCS: {fake_gcs_output_path}")
            return output_blob_name
        else:
            # Use real GCS
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(output_blob_name)
            blob.upload_from_filename(csv_path)
            
            logger.info(f"Uploaded CSV to gs://{bucket_name}/{output_blob_name}")
            return output_blob_name
        
    except Exception as e:
        logger.error(f"Error uploading CSV to GCS: {e}")
        raise


def process_pubsub_message(message_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process a Pub/Sub message and extract metrics from PDF"""
    try:
        # Extract bucket and file path from message
        bucket_name = message_data.get('bucket')
        file_path = message_data.get('name')
        
        if not bucket_name or not file_path:
            raise ValueError("Missing 'bucket' or 'name' in Pub/Sub message")
        
        logger.info(f"Processing file: gs://{bucket_name}/{file_path}")
        
        # Load metrics configuration
        metrics_config = load_metrics_config(METRICS_YAML_PATH)
        
        # Download PDF from GCS
        pdf_bytes = download_pdf_from_gcs(bucket_name, file_path)
        
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_bytes)
        logger.info(f"Extracted {len(text)} characters from PDF")
        
        # Extract metrics using langextract
        metrics = extract_metrics_from_text(text, metrics_config)
        
        # Generate CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            csv_path = tmp_file.name
        
        generate_csv(metrics, csv_path)
        
        # Upload CSV to GCS
        output_blob_name = upload_csv_to_gcs(bucket_name, csv_path, file_path)
        
        # Clean up temporary file
        os.unlink(csv_path)
        
        return {
            "status": "success",
            "input_file": f"gs://{bucket_name}/{file_path}",
            "output_file": f"gs://{bucket_name}/{output_blob_name}",
            "metrics_extracted": len(metrics),
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Error processing Pub/Sub message: {e}")
        raise


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
        
        result = process_pubsub_message(message_data)
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
        result = process_pubsub_message(message_data)
        return result
    except Exception as e:
        logger.error(f"Error in /process-direct endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ollama_model": OLLAMA_MODEL,
        "ollama_timeout_seconds": OLLAMA_TIMEOUT,
        "extract_max_chars": EXTRACT_MAX_CHARS or None,
        "use_fake_gcs": USE_FAKE_GCS,
        "use_fake_pubsub": USE_FAKE_PUBSUB,
        "fake_gcs_root": FAKE_GCS_ROOT if USE_FAKE_GCS else None
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
