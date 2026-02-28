from typing import Any, Dict, List
from langextract.core.data import ExampleData

class PromptManager:

    def create_examples_from_metrics(metrics_config: Dict[str, Any]) -> List[ExampleData]:
        from src.extraction.Extraction import Extraction
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

            # Ensure all attribute values are strings, integers, or floats (langextract requirement)
            # Convert list of synonyms to comma-separated string
            synonyms_str = ", ".join(synonyms) if isinstance(synonyms, list) else str(synonyms)

            # Create extraction
            extraction = Extraction(
                extraction_class=metric_name,
                extraction_text=example_value,
                attributes={
                    "metric_name": metric_name,
                    "value": example_value,
                    "definition": definition,
                    "synonyms": synonyms_str
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
