import json
from typing import Any


def _parse_ollama_json(content: str) -> Any:
    """Parse JSON from Ollama output, optionally stripping <think> tags."""
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        import re
        think_re = re.compile(r"<think>[\s\S]*?</think>\s*", re.IGNORECASE)
        if think_re.search(content):
            stripped = think_re.sub("", content).strip()
            return json.loads(stripped)
        raise