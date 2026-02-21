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
import re
import traceback
import ast

import yaml
from fastapi import FastAPI, HTTPException, Request
from google.cloud import storage
import PyPDF2
from pydantic import BaseModel
import langextract as lx
from langextract.core.data import ExampleData, Extraction
from .LenientOllamaFormatHandler import LenientOllamaFormatHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Document Extraction API")

# Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "1200"))  # per-request timeout (seconds)
METRICS_YAML_PATH = os.getenv("METRICS_YAML_PATH", "src/metrics.yaml")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "../data")
OUTPUT_FOLDER = "output"
# Limit text length for extraction to avoid timeouts on large docs (0 = no limit)
EXTRACT_MAX_CHARS = int(os.getenv("EXTRACT_MAX_CHARS", "5000"))

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

def _normalize_extraction_value(val: Any) -> Any:
    """Normalize a possibly-nested extraction value into a primitive (str/int/float).

    Handles:
    - primitives (return as-is)
    - lists (join with comma)
    - dicts (prefer 'value' or 'extraction_text' keys)
    - objects with attributes (Extraction-like)
    - stringified dicts or reprs of Extraction(...) - parse to extract inner value
    - fallback: str(val)
    """
    if val is None:
        return ""
    # If it's already a primitive, return
    if isinstance(val, (str, int, float, bool)):
        # If string looks like a dict or Extraction repr, try to parse inner value
        if isinstance(val, str):
            s = val.strip()
            # Heuristics: stringified dict starting with { or looks like "Extraction("
            if s.startswith('{') or 'extraction_text' in s or s.startswith('Extraction('):
                # Try ast.literal_eval for dict-like strings
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, dict):
                        # prefer keys
                        for key in ('value', 'extraction_text', 'text', 'extractionText'):
                            if key in parsed and parsed[key] is not None:
                                return _normalize_extraction_value(parsed[key])
                        # try attributes
                        attrs = parsed.get('attributes') if isinstance(parsed, dict) else None
                        if isinstance(attrs, dict) and attrs.get('value'):
                            return _normalize_extraction_value(attrs.get('value'))
                        # otherwise return a JSON string
                        try:
                            return json.dumps(parsed)
                        except Exception:
                            return str(parsed)
                except Exception:
                    # Try regex extraction from reprs like Extraction(... extraction_text='$1' ...)
                    m = re.search(r"extraction_text\s*[=:]\s*['\"]([^'\"]+)['\"]", s)
                    if m:
                        return m.group(1)
                    m2 = re.search(r"'value'\s*[:=]\s*['\"]([^'\"]+)['\"]", s)
                    if m2:
                        return m2.group(1)
                    # fallback to returning original string
                    return s
            return val
        return val
    # Lists -> join
    if isinstance(val, list):
        parts = [_normalize_extraction_value(v) for v in val]
        return ", ".join(str(p) for p in parts if p is not None)
    # Dict -> prefer values
    if isinstance(val, dict):
        for key in ("value", "extraction_text", "text", "extractionText"):
            if key in val and val[key] is not None:
                return _normalize_extraction_value(val[key])
        for v in val.values():
            if isinstance(v, (str, int, float)):
                return v
        try:
            return json.dumps(val)
        except Exception:
            return str(val)
    # Objects
    try:
        if hasattr(val, 'extraction_text'):
            return _normalize_extraction_value(getattr(val, 'extraction_text'))
        if hasattr(val, 'attributes') and getattr(val, 'attributes'):
            attrs = getattr(val, 'attributes')
            if isinstance(attrs, dict) and attrs.get('value'):
                return _normalize_extraction_value(attrs.get('value'))
        if hasattr(val, 'value'):
            return _normalize_extraction_value(getattr(val, 'value'))
    except Exception:
        pass
    return str(val)

def getExtractedMetrics(result):
    """Normalize LangExtract/Extraction output into a list of simple metric dicts.

    Handles multiple shapes returned by the resolver:
    - Extraction objects (with .extraction_text, .extraction_class, .attributes)
    - dicts that look like extraction records
    - cases where the LLM put the actual value into attributes['value'] instead of extraction_text
    """
    extracted_metrics = []

    for extraction in result.extractions:
        # Initialize placeholders
        metric_name = ""
        val: Any = ""
        description = ""

        # Case A: extraction is a dict-like object
        if isinstance(extraction, dict):
            metric_name = extraction.get('extraction_class') or extraction.get('metric_name') or ""
            # Prefer attributes.value
            attrs = extraction.get('attributes') or {}
            if isinstance(attrs, dict) and attrs.get('value'):
                val = attrs.get('value')
            else:
                val = extraction.get('extraction_text')
            description = extraction.get('description') or attrs.get('description', '')
        else:
            # Case B: object with attributes (langextract Extraction)
            metric_name = getattr(extraction, 'extraction_class', '') or ''
            # Try attributes first
            attrs = getattr(extraction, 'attributes', None)
            if isinstance(attrs, dict) and attrs.get('value'):
                val = attrs.get('value')
            else:
                val = getattr(extraction, 'extraction_text', '')
            description = getattr(extraction, 'description', '') or (attrs.get('description') if isinstance(attrs, dict) else '')

        # If metric_name empty, try to pull from attributes
        if not metric_name and isinstance(attrs, dict):
            metric_name = attrs.get('metric_name', '') or attrs.get('name', '')

        # Coerce None to empty string
        if val is None:
            val = ""

        # Normalize metric_name if it's a complex object or repr
        if metric_name is None:
            metric_name = ""
        if not isinstance(metric_name, (str, int, float, bool)):
            metric_name = _normalize_extraction_value(metric_name)
        else:
            # If metric_name is a string but looks like a repr, try to extract a sane name
            if isinstance(metric_name, str):
                mn = metric_name.strip()
                # Common pattern: "Extraction(extraction_class='nav_per_share', ... )"
                m = re.search(r"extraction_class\s*=\s*['\"]([a-zA-Z0-9_\-]+)['\"]", mn)
                if m:
                    metric_name = m.group(1)
                # Or stringified dict
                elif mn.startswith('{'):
                    try:
                        parsed_mn = ast.literal_eval(mn)
                        if isinstance(parsed_mn, dict):
                            metric_name = parsed_mn.get('extraction_class') or parsed_mn.get('metric_name') or metric_name
                    except Exception:
                        pass

        # If value is a non-primitive (list/dict/object), convert to a concise string
        val = _normalize_extraction_value(val)

        # Normalize the extracted value into a primitive
        val = _normalize_extraction_value(val)

        # If val is a string that still contains a nested dict/repr, try to extract inner value
        if isinstance(val, str):
            s = val.strip()
            if s.startswith('{') or 'extraction_text' in s or s.startswith('Extraction('):
                # try to extract inner value
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, dict):
                        for key in ('value', 'extraction_text', 'text'):
                            if key in parsed and parsed[key]:
                                val = _normalize_extraction_value(parsed[key])
                                break
                        else:
                            # Try attributes
                            attrs_p = parsed.get('attributes')
                            if isinstance(attrs_p, dict) and attrs_p.get('value'):
                                val = _normalize_extraction_value(attrs_p.get('value'))
                except Exception:
                    m = re.search(r"extraction_text\s*[=:]\s*['\"]([^'\"]+)['\"]", s)
                    if m:
                        val = m.group(1)

        # Final ensure primitive string formatting
        if isinstance(val, bool):
            val = str(val)
        elif isinstance(val, (int, float)):
            val = val
        else:
            val = str(val).strip()

        # Only include non-empty values (keep empty strings if metric was found but empty?)
        # We'll include metrics even if empty so CSV has rows; caller can filter later.
        metric_data = {
            "metric_name": metric_name or "",
            "value": val,
            "description": description or ""
        }
        extracted_metrics.append(metric_data)

    return extracted_metrics

def rule_based_extract(text: str, metrics_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fallback extractor that searches the document for metric mentions using regex heuristics.

    This is intended as a robust backup when the LLM output cannot be parsed into JSON.
    It looks for currency, percentages, and metric-specific keywords/synonyms and returns
    a list of metric dictionaries with metric_name, value, and description (context).
    """
    results: List[Dict[str, Any]] = []
    if not text:
        return results

    # Normalize whitespace
    txt = re.sub(r"\s+", " ", text)

    # Common patterns
    currency_re = re.compile(r"(?:(?:\$|USD\s?)\s?[0-9]{1,3}(?:[,\s][0-9]{3})*(?:\.[0-9]+)?|[0-9]{1,3}(?:[,\s][0-9]{3})+\s?(?:USD|usd|dollars|\$))")
    percent_re = re.compile(r"-?\d{1,3}(?:\.\d+)?%")
    decimal_re = re.compile(r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?")

    # Helper to grab context snippet around a match
    def snippet(match_span):
        start, end = match_span
        s = max(0, start - 50)
        e = min(len(txt), end + 50)
        return txt[s:e].strip()

    # Find all numeric tokens and their spans to allow closest-match selection
    numeric_tokens = []
    for m in currency_re.finditer(txt):
        numeric_tokens.append((m.start(), m.end(), m.group(0)))
    for m in percent_re.finditer(txt):
        numeric_tokens.append((m.start(), m.end(), m.group(0)))
    for m in decimal_re.finditer(txt):
        numeric_tokens.append((m.start(), m.end(), m.group(0)))
    # Sort tokens by start
    numeric_tokens.sort(key=lambda x: x[0])

    # Track used numeric spans so we don't reuse the same value for multiple metrics
    used_spans: List[tuple] = []

    def span_used(start: int, end: int) -> bool:
        for (rs, re) in used_spans:
            if not (end <= rs or start >= re):
                return True
        return False

    def mark_used(start: int, end: int):
        used_spans.append((start, end))

    metrics = metrics_config.get('metrics', [])
    lowered = txt.lower()

    # Pre-check label patterns per metric (more reliable)
    label_patterns = {
        'fund_aum': re.compile(r"assets under management[:\s]*([\$\d,\.MKBbn]+)", re.IGNORECASE),
        'nav_per_share': re.compile(r"nav(?: per share)?[:\s]*([\$\d,\.]+)", re.IGNORECASE),
        'quarterly_return': re.compile(r"quarter(?:ly)? return[:\s]*(-?\d{1,3}(?:\.\d+)?%)", re.IGNORECASE),
        'ytd_return': re.compile(r"year[- ]to[- ]date return[:\s]*(-?\d{1,3}(?:\.\d+)?%)", re.IGNORECASE),
        'one_year_return': re.compile(r"(one[- ]year|12[- ]month) return[:\s]*(-?\d{1,3}(?:\.\d+)?%)", re.IGNORECASE),
        'expense_ratio': re.compile(r"expense ratio[:\s]*([\d\.]+%)", re.IGNORECASE),
        'net_flows': re.compile(r"net flows[:\s]*([\$\d,\.]+)", re.IGNORECASE),
        'turnover_ratio': re.compile(r"turnover(?: ratio)?[:\s]*([\d\.]+%)", re.IGNORECASE),
        'max_drawdown': re.compile(r"max(?:imum)? drawdown[:\s]*(-?\d{1,3}(?:\.\d+)?%)", re.IGNORECASE),
    }

    for metric in metrics:
        name = metric.get('name')
        synonyms = metric.get('synonyms', []) or []
        found = False

        # 1) Try label-specific patterns first
        pat = label_patterns.get(name)
        if pat:
            m = pat.search(txt)
            if m:
                # prefer first capturing group with numeric value
                val = m.group(1) if m.groups() else m.group(0)
                # Determine start/end span safely when group exists
                if m.lastindex and m.lastindex >= 1:
                    start = m.start(1)
                    end = m.end(1)
                else:
                    start = m.start()
                    end = m.end()
                # mark span used
                try:
                    mark_used(start, end)
                except Exception:
                    pass
                results.append({
                    'metric_name': name,
                    'value': val.strip(),
                    'description': snippet((m.start(), m.end()))
                })
                continue

        # 2) Keyword-based search: look for keyword occurrences and choose the closest numeric token
        keywords = [name.replace('_', ' ')] + [s for s in synonyms]
        for kw in keywords:
            if not kw:
                continue
            for m_kw in re.finditer(re.escape(kw), txt, re.IGNORECASE):
                kw_center = (m_kw.start() + m_kw.end()) // 2
                # find closest numeric token by distance to kw_center
                best = None
                best_dist = None
                for (s, e, tok) in numeric_tokens:
                    if span_used(s, e):
                        continue
                    # distance from kw_center to token span
                    dist = min(abs(kw_center - s), abs(kw_center - e))
                    if best is None or dist < best_dist:
                        best = (s, e, tok)
                        best_dist = dist
                if best:
                    s, e, tok = best
                    mark_used(s, e)
                    results.append({
                        'metric_name': name,
                        'value': tok.strip(),
                        'description': snippet((s, e))
                    })
                    found = True
                    break
            if found:
                break

    # Deduplicate by metric_name keeping the first occurrence
    final = {}
    for r in results:
        if r['metric_name'] not in final:
            final[r['metric_name']] = r
    return list(final.values())


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
                    "format_handler": LenientOllamaFormatHandler()
                }
            )

            # Convert extractions to list of dictionaries
            extracted_metrics = []
            if isinstance(result, list):
                for doc in result:
                    if hasattr(doc, 'extractions') and doc.extractions:
                        extracted_metrics = getExtractedMetrics(doc)
            elif hasattr(result, 'extractions') and result.extractions:
                extracted_metrics = getExtractedMetrics(result)

            logger.info(f"Extracted {len(extracted_metrics)} metrics")
            return extracted_metrics
        finally:
            resolver_module.Resolver.extract_ordered_extractions = original_extract

    except Exception as e:
        # Log full traceback for debugging
        tb = traceback.format_exc()
        logger.error(f"Error extracting metrics: {e}\n{tb}")
        logger.info("Falling back to rule-based extraction")
        try:
            fallback = rule_based_extract(text, metrics_config)
            logger.info(f"Rule-based extractor found {len(fallback)} metrics")
            return fallback
        except Exception as e2:
            # If fallback fails, log and return empty list instead of raising to avoid 500
            tb2 = traceback.format_exc()
            logger.error(f"Fallback extraction failed: {e2}\n{tb2}")
            logger.warning("Returning empty metrics list due to extraction failures")
            return []

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
        try:
            metrics = extract_metrics_from_text(text, metrics_config)
        except Exception as e:
            # Extraction failed unexpectedly despite internal fallbacks; log and continue with empty metrics
            logger.error(f"Metric extraction failed in process_pubsub_message: {e}")
            metrics = []

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
