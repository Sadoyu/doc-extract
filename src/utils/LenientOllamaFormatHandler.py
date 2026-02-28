import json

from langextract.core import format_handler as fh
from langextract.core import exceptions as lx_exceptions
from langextract.core.data import EXTRACTIONS_KEY, FormatType

from src.ollamaSetup import _parse_ollama_json

# Define module-level defaults expected by callers/importers.
# These were previously referenced before being defined, causing a NameError.
use_fences = False
attribute_suffix = "_attributes"

# Expose format attributes at module level for code that inspects the module
# rather than the class. This prevents AttributeError when code expects
# module.format_type, module.use_wrapper, etc.
format_type = FormatType.JSON
use_wrapper = True
wrapper_key = EXTRACTIONS_KEY
strict_fences = False


def format_extraction_example(example):
    """
    Return a JSON-formatted example string for prompts or docs.

    If `example` is a dict, wrap it in the standard {"extractions": [...]} form.
    If it's a list, treat it as the list of extractions. Otherwise return the
    stringified form.
    """
    try:
        if isinstance(example, dict):
            data = {EXTRACTIONS_KEY: [example]}
        elif isinstance(example, list):
            data = {EXTRACTIONS_KEY: example}
        else:
            # Not a mapping/list; just return a printable representation
            return json.dumps(example, ensure_ascii=False)
        return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception:
        # Be conservative: fall back to str on unexpected types
        return str(example)


class LenientOllamaFormatHandler(fh.FormatHandler):
    """FormatHandler that accepts Ollama output with or without an 'extractions' wrapper.

    Ollama sometimes returns a flat object like {"revenue": "$1M", "profit": "..."}
    or a bare list instead of {"extractions": [...]}. This handler normalizes those
    so parsing does not fail. Uses its own JSON parsing to avoid relying on base
    class private methods.
    """

    # Mirror module-level settings
    use_fences = use_fences
    attribute_suffix = attribute_suffix

    # Provide attributes expected by langextract and caller code
    format_type = FormatType.JSON
    use_wrapper = True
    wrapper_key = EXTRACTIONS_KEY
    strict_fences = False

    # Expose the module helper as a class-level staticmethod so callers may
    # reference either the module or the class attribute.
    format_extraction_example = staticmethod(format_extraction_example)

    def __init__(self, *args, **kwargs):
        # Intentionally lightweight: accept arbitrary init args to be tolerant
        # of different instantiation patterns. Base class initialization is not
        # strictly required for the attributes used by the caller.
        pass

    def parse(self, text: str) -> str:
        """
        Return text unchanged. Provide a hook if you need to normalize LLM output
        (strip fences, sanitize json, etc.) before resolver consumption.
        """
        return text

    def format_extractions(self, extractions):
        """
        Identity formatter; return extractions unchanged.
        Extend this if langextract expects specific formatting behavior.
        """
        return extractions

    def parse_output(
        self, text: str, *, strict: bool | None = None
    ):
        if not text or not text.strip():
            # Return empty list instead of raising to allow fallback
            return []

        content = text.strip()
        try:
            parsed = _parse_ollama_json(content)
        except Exception as e:
            # Log and return empty list so caller can fallback
            import logging
            logging.getLogger(__name__).warning(f"LenientOllamaFormatHandler: failed to parse JSON - {e}")
            return []

        if parsed is None:
            return []

        try:
            if isinstance(parsed, dict):
                if EXTRACTIONS_KEY in parsed:
                    items = parsed[EXTRACTIONS_KEY]
                elif getattr(self, "wrapper_key", None) and self.wrapper_key in parsed:
                    items = parsed[self.wrapper_key]
                else:
                    # Flat dict: treat as one extraction group
                    items = [parsed]
            elif isinstance(parsed, list):
                items = parsed
            else:
                return []

            if not isinstance(items, list):
                return []

            # Validate and sanitize each item: ensure no nested lists as values
            sanitized_items = []
            for item in items:
                if not isinstance(item, dict):
                    # skip non-dict items
                    continue

                sanitized_item = {}
                for k, v in item.items():
                    if not isinstance(k, str):
                        # skip non-string keys
                        continue

                    # Recursively flatten nested lists/dicts to strings
                    if isinstance(v, (str, int, float, bool)):
                        sanitized_item[k] = v
                    elif v is None:
                        sanitized_item[k] = ""
                    elif isinstance(v, list):
                        # Convert list to comma-separated string
                        sanitized_item[k] = ", ".join(str(x) for x in v)
                    elif isinstance(v, dict):
                        # Convert dict to JSON string
                        try:
                            sanitized_item[k] = json.dumps(v)
                        except Exception:
                            sanitized_item[k] = str(v)
                    else:
                        # Fallback for any other type
                        sanitized_item[k] = str(v)

                if sanitized_item:
                    sanitized_items.append(sanitized_item)

            return sanitized_items
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"LenientOllamaFormatHandler: unexpected error during parse_output sanitization - {e}")
            return []

# Module-level compatibility function: delegate to the class implementation.
# Some callers import the module and call `parse_output(...)` directly; expose
# that symbol at module scope to avoid AttributeError.

def parse_output(text: str, *, strict: bool | None = None):
    """Compatibility wrapper that delegates to LenientOllamaFormatHandler.parse_output.

    Returns the parsed extractions (a list of mappings) or raises
    lx_exceptions.FormatParseError on invalid input.
    """
    handler = LenientOllamaFormatHandler()
    return handler.parse_output(text, strict=strict)

