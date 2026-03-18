"""Context firewall — sanitize retrieved document text before LLM prompt assembly.

Mitigates OWASP LLM01 (prompt injection) by replacing known injection trigger
phrases with a neutral marker.  This is a defence-in-depth layer; it does not
replace the anti-injection system prompt directive in config/defaults.yaml.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Patterns that commonly appear in prompt-injection payloads embedded in docs.
# All are matched case-insensitively.  re.MULTILINE makes ^ match at the start
# of every line, which is used to anchor patterns that would otherwise produce
# false positives mid-sentence (e.g. "operating system: Linux" or
# "services act as brokers").
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE | re.MULTILINE)
    for p in [
        r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
        r"you\s+are\s+now\s+",
        r"^\s*act\s+as\s+(a\s+)?",  # start-of-line: avoids "X acts as Y" mid-sentence
        r"disregard\s+(all\s+)?(previous|prior)\s+",
        r"new\s+instructions?\s*:",
        r"^\s*system\s*:",  # start-of-line: avoids "operating system: Linux"
        r"<\s*/?system\s*>",
        r"\[\s*system\s*\]",
    ]
]

_REDACTION_MARKER = "[REDACTED]"


def sanitize_document_text(text: str) -> str:
    """Replace injection trigger phrases in *text* with a redaction marker.

    Returns the sanitized string.  If no triggers are found the original string
    is returned unchanged (no allocation overhead for the common case).
    """
    sanitized = text
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(sanitized):
            logger.warning(
                "Context firewall: injection trigger matched by pattern %r — redacting",
                pattern.pattern,
            )
            sanitized = pattern.sub(_REDACTION_MARKER, sanitized)
    return sanitized
