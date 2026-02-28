from typing import Any, Dict
import yaml, os, logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:

    # Configuration
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "1200"))  # per-request timeout (seconds)
    METRICS_YAML_PATH = os.getenv("METRICS_YAML_PATH", "src/metrics.yaml")
    GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "../data")
    OUTPUT_FOLDER = "output"
    # Limit text length for extraction to avoid timeouts on large docs (0 = no limit)
    EXTRACT_MAX_CHARS = int(os.getenv("EXTRACT_MAX_CHARS", "0"))

    # Fake GCS/PubSub Configuration
    USE_FAKE_GCS = os.getenv("USE_FAKE_GCS", "true").lower() == "true"
    FAKE_GCS_ROOT = os.getenv("FAKE_GCS_ROOT", "/Users/saikiranchandolu/Workspace/data-bucket")
    USE_FAKE_PUBSUB = os.getenv("USE_FAKE_PUBSUB", "true").lower() == "true"

    def load_metrics_config(yaml_path: str) -> Dict[str, Any]:
        """Load metrics configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)
