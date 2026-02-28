from typing import Any, Dict, List
import ast, logging, re, json, traceback, PyPDF2, tempfile
import langextract as lx
from src.config.Config import Config
from src.utils.LenientOllamaFormatHandler import LenientOllamaFormatHandler
from src.prompts.PromptManager import PromptManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Extraction:

    def __init__(self, extraction_class=None, **kwargs):
        self.extraction_class = extraction_class
        # This captures attributes, extraction_text, and anything else automatically
        for key, value in kwargs.items():
            setattr(self, key, value)

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
                description = getattr(extraction, 'description', '') or (
                    attrs.get('description') if isinstance(attrs, dict) else '')

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
                metric_name = Extraction._normalize_extraction_value(metric_name)
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
                                metric_name = parsed_mn.get('extraction_class') or parsed_mn.get(
                                    'metric_name') or metric_name
                        except Exception:
                            pass

            # If value is a non-primitive (list/dict/object), convert to a concise string
            val = Extraction._normalize_extraction_value(val)

            # Normalize the extracted value into a primitive
            val = Extraction._normalize_extraction_value(val)

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
                                    val = Extraction._normalize_extraction_value(parsed[key])
                                    break
                            else:
                                # Try attributes
                                attrs_p = parsed.get('attributes')
                                if isinstance(attrs_p, dict) and attrs_p.get('value'):
                                    val = Extraction._normalize_extraction_value(attrs_p.get('value'))
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
        currency_re = re.compile(
            r"(?:(?:\$|USD\s?)\s?[0-9]{1,3}(?:[,\s][0-9]{3})*(?:\.[0-9]+)?|[0-9]{1,3}(?:[,\s][0-9]{3})+\s?(?:USD|usd|dollars|\$))")
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

        # Pre-check label patterns per metric (more reliable)
        label_patterns = {
            'fund_aum': re.compile(r"assets under management[:\s]*([\$\d,\.MKBbn]+)", re.IGNORECASE),
            'nav_per_share': re.compile(r"nav(?: per share)?[:\s]*([\$\d,\.]+)", re.IGNORECASE),
            'quarterly_return': re.compile(r"quarter(?:ly)? return[:\s]*(-?\d{1,3}(?:\.\d+)?%)", re.IGNORECASE),
            'ytd_return': re.compile(r"year[- ]to[- ]date return[:\s]*(-?\d{1,3}(?:\.\d+)?%)", re.IGNORECASE),
            'one_year_return': re.compile(r"(one[- ]year|12[- ]month) return[:\s]*(-?\d{1,3}(?:\.\d+)?%)",
                                          re.IGNORECASE),
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
            if Config.EXTRACT_MAX_CHARS and len(text) > Config.EXTRACT_MAX_CHARS:
                text = text[:Config.EXTRACT_MAX_CHARS]
                logger.info(f"Truncated text to {Config.EXTRACT_MAX_CHARS} chars for extraction")

            # Create examples and prompt
            examples = PromptManager.create_examples_from_metrics(metrics_config)
            prompt_description = PromptManager.build_prompt_description(metrics_config)

            logger.info(f"Extracting metrics using Ollama model: {Config.OLLAMA_MODEL} (timeout={Config.OLLAMA_TIMEOUT}s)")

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
                sanitized = Extraction._sanitize_extraction_data_for_resolver(
                    extraction_data, index_suffix, attributes_suffix
                )
                return original_extract(self, sanitized)

            resolver_module.Resolver.extract_ordered_extractions = patched_extract_ordered_extractions
            try:
                result = lx.extract(
                    text_or_documents=text,
                    model_id=Config.OLLAMA_MODEL,
                    prompt_description=prompt_description,
                    examples=examples,
                    max_char_buffer=4000,  # Larger chunks = fewer Ollama calls, less timeout risk
                    extraction_passes=1,  # Single pass to reduce total time
                    max_workers=2,  # Fewer parallel calls to avoid overloading local Ollama
                    language_model_params={
                        "timeout": Config.OLLAMA_TIMEOUT,
                        "model_url": Config.OLLAMA_URL
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
                            extracted_metrics = Extraction.getExtractedMetrics(doc)
                elif hasattr(result, 'extractions') and result.extractions:
                    extracted_metrics = Extraction.getExtractedMetrics(result)

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
                fallback = Extraction.rule_based_extract(text, metrics_config)
                logger.info(f"Rule-based extractor found {len(fallback)} metrics")
                return fallback
            except Exception as e2:
                # If fallback fails, log and return empty list instead of raising to avoid 500
                tb2 = traceback.format_exc()
                logger.error(f"Fallback extraction failed: {e2}\n{tb2}")
                logger.warning("Returning empty metrics list due to extraction failures")
                return []

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
                                    return Extraction._normalize_extraction_value(parsed[key])
                            # try attributes
                            attrs = parsed.get('attributes') if isinstance(parsed, dict) else None
                            if isinstance(attrs, dict) and attrs.get('value'):
                                return Extraction._normalize_extraction_value(attrs.get('value'))
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
            parts = [Extraction._normalize_extraction_value(v) for v in val]
            return ", ".join(str(p) for p in parts if p is not None)
        # Dict -> prefer values
        if isinstance(val, dict):
            for key in ("value", "extraction_text", "text", "extractionText"):
                if key in val and val[key] is not None:
                    return Extraction._normalize_extraction_value(val[key])
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
                return Extraction._normalize_extraction_value(getattr(val, 'extraction_text'))
            if hasattr(val, 'attributes') and getattr(val, 'attributes'):
                attrs = getattr(val, 'attributes')
                if isinstance(attrs, dict) and attrs.get('value'):
                    return Extraction._normalize_extraction_value(attrs.get('value'))
            if hasattr(val, 'value'):
                return Extraction._normalize_extraction_value(getattr(val, 'value'))
        except Exception:
            pass
        return str(val)

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
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise

        return "\n".join(text_content)

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
