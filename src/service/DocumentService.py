from src.config.Config import Config
from src.storage.StorageAdapter import StorageAdapter
from src.extraction import Extraction
from src.utils.CSVGenerator import CSVGenerator
import tempfile, os, traceback
from typing import Any, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentService:

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
            metrics_config = Config.load_metrics_config(Config.METRICS_YAML_PATH)

            # Download PDF from GCS
            pdf_bytes = StorageAdapter.download_pdf_from_gcs(bucket_name, file_path)

            # Extract text from PDF
            text = Extraction.extract_text_from_pdf(pdf_bytes)
            logger.info(f"Extracted {len(text)} characters from PDF")

            # Extract metrics using langextract
            try:
                metrics = Extraction.extract_metrics_from_text(text, metrics_config)
            except Exception as e:
                # Extraction failed unexpectedly despite internal fallbacks; log and continue with empty metrics
                logger.error(f"Metric extraction failed in process_pubsub_message: {e}")
                metrics = []

            # Generate CSV file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                csv_path = tmp_file.name

            CSVGenerator.generate_csv(metrics, csv_path)

            # Upload CSV to GCS
            output_blob_name = StorageAdapter.upload_csv_to_gcs(bucket_name, csv_path, file_path)

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
            tb = traceback.format_exc()
            logger.error(f"Error processing Pub/Sub message: {e}\n{tb}")
            # Return a structured error so HTTP handlers can respond without raising
            return {
                "status": "error",
                "error": str(e),
                "traceback": tb,
                "input_file": f"gs://{message_data.get('bucket')}/{message_data.get('name')}",
                "metrics_extracted": 0,
                "metrics": []
            }
