import src.config.Config as Config
import logging
from google.cloud import storage
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StorageAdapter:

    def download_pdf_from_gcs(bucket_name: str, file_path: str) -> bytes:
        """Download PDF file from GCS bucket (real or fake)"""
        try:
            if Config.USE_FAKE_GCS:
                # Use local filesystem as fake GCS
                fake_gcs_path = Path(Config.FAKE_GCS_ROOT) / bucket_name / file_path
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

    def upload_csv_to_gcs(bucket_name: str, csv_path: str, original_file_path: str):
        """Upload CSV file to GCS bucket in output folder (real or fake)"""
        try:
            # Create output path: output/filename.csv
            original_filename = Path(original_file_path).stem
            output_blob_name = f"{Config.OUTPUT_FOLDER}/{original_filename}_metrics.csv"

            if Config.USE_FAKE_GCS:
                # Use local filesystem as fake GCS
                fake_gcs_output_path = Path(Config.FAKE_GCS_ROOT) / bucket_name / output_blob_name
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
