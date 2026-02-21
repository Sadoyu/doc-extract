import json

from langextract.core import format_handler as fh
from langextract.core import exceptions as lx_exceptions
from langextract.core.data import EXTRACTIONS_KEY, FormatType

from ollamaSetup import _parse_ollama_json


class LenientOllamaFormatHandler(fh.FormatHandler):
    """FormatHandler that accepts Ollama output with or without an 'extractions' wrapper.

    Ollama sometimes returns a flat object like {"revenue": "$1M", "profit": "..."}
    or a bare list instead of {"extractions": [...]}. This handler normalizes those
    so parsing does not fail. Uses its own JSON parsing to avoid relying on base
    class private methods.
    """

    def __init__(self):
        super().__init__()
        self.format_type = FormatType.JSON,
        self.use_wrapper = True,
        self.wrapper_key = EXTRACTIONS_KEY,
        self.use_fences = False,
        self.strict_fences = False,

    def parse_output(
        self, text: str, *, strict: bool | None = None
    ):
        if not text or not text.strip():
            raise lx_exceptions.FormatParseError("Empty or invalid input string.")

        content = text.strip()
        try:
            parsed = _parse_ollama_json(content)
        except json.JSONDecodeError as e:
            raise lx_exceptions.FormatParseError(
                f"Failed to parse JSON content: {e!s}"[:200]
            ) from e

        if parsed is None:
            raise lx_exceptions.FormatParseError(
                "Content must be a mapping or a list of mappings."
            )

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
            raise lx_exceptions.FormatParseError(
                f"Expected list or dict, got {type(parsed)}"
            )

        if not isinstance(items, list):
            raise lx_exceptions.FormatParseError(
                "The extractions must be a sequence (list) of mappings."
            )

        for item in items:
            if not isinstance(item, dict):
                raise lx_exceptions.FormatParseError(
                    "Each item in the sequence must be a mapping."
                )
            for k in item.keys():
                if not isinstance(k, str):
                    raise lx_exceptions.FormatParseError(
                        "All extraction keys must be strings (got a non-string key)."
                    )

        return items
