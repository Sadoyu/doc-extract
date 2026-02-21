import json
import re
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _parse_ollama_json(content: str) -> Any:
    """
    Parse JSON from Ollama output with lenient handling for common LLM JSON errors.

    Handles:
    - <think> tags (strip them)
    - Trailing commas in objects/arrays
    - Unquoted keys
    - Missing commas between properties
    - Single quotes instead of double quotes
    - Comments
    - Unterminated strings
    - Extra whitespace/newlines
    - Incomplete/truncated JSON
    """
    content = content.strip()

    # Try direct parse first (fastest path)
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.debug(f"Direct JSON parse failed at position {e.pos}: {e.msg}")

    # Remove <think> tags
    think_re = re.compile(r"<think>[\s\S]*?</think>\s*", re.IGNORECASE)
    if think_re.search(content):
        content = think_re.sub("", content).strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.debug(f"Parse after removing <think> tags failed: {e.msg}")

    # Try to fix common JSON issues
    try:
        return _repair_and_parse_json(content)
    except Exception as e:
        logger.error(f"Failed to repair JSON: {e}")
        # Last resort: try to extract key-value pairs as a simple dict
        try:
            return _extract_key_values(content)
        except Exception as e2:
            logger.error(f"Failed to extract key-values: {e2}")
            raise json.JSONDecodeError(
                f"Could not parse JSON after all repair attempts",
                content,
                0
            )


def _repair_and_parse_json(content: str) -> Any:
    """
    Attempt to repair common JSON formatting issues from LLM output.
    """
    # Remove markdown code blocks if present
    content = re.sub(r'^```json\s*', '', content)
    content = re.sub(r'^```\s*', '', content)
    content = re.sub(r'\s*```$', '', content)
    content = content.strip()

    # Fix unterminated strings: if we have an unterminated string at the end,
    # close it before the last character
    content = _fix_unterminated_strings(content)

    # Replace single quotes with double quotes (but be careful with apostrophes)
    # This is a simplistic approach - only at start of strings
    content = re.sub(r":\s*'([^']*)'", r': "\1"', content)
    content = re.sub(r",\s*'([^']*)'", r', "\1"', content)
    content = re.sub(r"\[\s*'([^']*)'", r'["\1"', content)

    # Fix unquoted keys: "key": value -> "key": value
    # Match pattern: {whitespace}word{whitespace}:
    content = re.sub(r'{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'{"\1":', content)
    content = re.sub(r',\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r',"\1":', content)

    # Remove trailing commas before } or ]
    content = re.sub(r',(\s*[}\]])', r'\1', content)

    # Remove comments (// style and # style)
    lines = content.split('\n')
    cleaned_lines = []
    for line in lines:
        # Remove // comments
        if '//' in line:
            line = line[:line.index('//')]
        # Remove # comments
        if '#' in line and not '"' in line[:line.index('#')]:
            line = line[:line.index('#')]
        cleaned_lines.append(line)
    content = '\n'.join(cleaned_lines)

    # Fix missing commas between properties
    # Pattern: }newline" -> },"
    content = re.sub(r'"\s*\n\s*"', '",\n"', content)

    # Fix escaped quotes that break JSON parsing
    # Replace \" at boundaries with actual quote handling
    content = _fix_escaped_quotes(content)

    # Ensure content starts with { or [
    content = content.strip()
    if not content.startswith(('{', '[')):
        raise ValueError(f"JSON must start with {{ or [, got: {content[:20]}")

    # Close any unclosed braces/brackets
    content = _close_unclosed_braces(content)

    # Try parsing after repairs
    try:
        result = json.loads(content)
        logger.debug("Successfully parsed JSON after repair")
        return result
    except json.JSONDecodeError as e:
        logger.debug(f"Parse after repair failed: {e.msg} at position {e.pos}")
        logger.debug(f"Content around error: ...{content[max(0, e.pos-50):min(len(content), e.pos+50)]}...")
        raise


def _fix_unterminated_strings(content: str) -> str:
    """
    Fix unterminated strings by finding and closing them.
    """
    # Find the last quote in the content
    # If the number of unescaped quotes is odd, close the last string

    # Count unescaped quotes
    i = 0
    quote_count = 0
    while i < len(content):
        if content[i] == '"' and (i == 0 or content[i-1] != '\\'):
            quote_count += 1
        i += 1

    # If odd number of quotes, add a closing quote
    if quote_count % 2 == 1:
        # Find where the last unterminated string starts
        i = len(content) - 1
        while i >= 0:
            if content[i] == '"' and (i == 0 or content[i-1] != '\\'):
                # Check if this quote is already closed
                remaining = content[i+1:]
                if remaining.count('"') % 2 == 0:
                    # This quote is unclosed, close it
                    # Find the end of this string
                    close_pos = len(content)
                    # If next char is a special char that shouldn't be in a string, truncate there
                    for j in range(i+1, len(content)):
                        if content[j] in '\n\r' and (j == 0 or content[j-1] != '\\'):
                            close_pos = j
                            break
                    content = content[:close_pos] + '"'
                    break
            i -= 1

    return content


def _fix_escaped_quotes(content: str) -> str:
    """
    Handle escaped quotes properly without breaking JSON parsing.
    """
    # This is tricky - we want to preserve valid escape sequences
    # but fix malformed ones

    # Replace \\" with \" (double-escaped quotes)
    content = content.replace('\\\\"', '\\"')

    return content


def _close_unclosed_braces(content: str) -> str:
    """
    Close any unclosed braces and brackets.
    """
    open_braces = 0
    open_brackets = 0
    in_string = False
    i = 0

    while i < len(content):
        if content[i] == '"' and (i == 0 or content[i-1] != '\\'):
            in_string = not in_string
        elif not in_string:
            if content[i] == '{':
                open_braces += 1
            elif content[i] == '}':
                open_braces -= 1
            elif content[i] == '[':
                open_brackets += 1
            elif content[i] == ']':
                open_brackets -= 1
        i += 1

    # Close unclosed braces and brackets
    content += '}' * open_braces + ']' * open_brackets

    return content


def _extract_key_values(content: str) -> Any:
    """
    Last-resort parser: extract key-value pairs using regex.
    Returns a dict of extracted values.
    """
    logger.warning("Using fallback key-value extraction (JSON was too malformed)")

    # Find all quoted strings and their values
    # Pattern: "key": value or 'key': value
    pattern = r'["\']([^"\']+)["\']\s*:\s*["\']?([^,}\]"\'\n]+)["\']?'

    matches = re.findall(pattern, content)
    if not matches:
        raise ValueError("Could not extract any key-value pairs from content")

    result = {}
    for key, value in matches:
        result[key.strip()] = value.strip()

    # Wrap in extractions format if we got results
    if result:
        return {"extractions": [result]}

    raise ValueError("Extracted key-values but result is empty")
