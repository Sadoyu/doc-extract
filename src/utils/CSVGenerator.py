from typing import Any, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSVGenerator:

    def generate_csv(metrics: List[Dict[str, Any]], output_path: str):
        """Generate CSV file from extracted metrics"""
        import csv

        # Always write header even if there are no metrics
        if not metrics:
            logger.warning("No metrics to write to CSV; creating header-only CSV")

        # Define consistent fieldnames
        fieldnames = ["metric_name", "value", "description"]

        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()

            for metric in metrics:
                row = {
                    "metric_name": metric.get("metric_name", ""),
                    "value": metric.get("value", ""),
                    "description": metric.get("description", "")
                }
                writer.writerow(row)

        logger.info(f"Generated CSV file: {output_path}")
