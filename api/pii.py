"""
PII redaction for stored chat content.

Conservative regex-based scrubbing — runs before any user-supplied text
is persisted to MongoDB. The store is meant for analytics / audit; the
LLM response itself is not affected.

Patterns covered:
- Email addresses
- Credit-card-like 13–19-digit sequences (with separators)
- US SSN (xxx-xx-xxxx)
- Phone numbers (international + US/EU formats)

Intentionally narrow — false positives are preferred over leaking PII.
"""

from __future__ import annotations

import re

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_CC_RE    = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
_SSN_RE   = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE_RE = re.compile(
    r"(?:(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}(?!\d))"
)


def redact_pii(text: str | None) -> str:
    """Return ``text`` with emails / phones / SSNs / credit-cards replaced
    by ``[REDACTED_*]`` markers. Empty / None input passes through.
    """
    if not text:
        return text or ""
    out = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    out = _CC_RE.sub("[REDACTED_CC]", out)
    out = _SSN_RE.sub("[REDACTED_SSN]", out)
    out = _PHONE_RE.sub("[REDACTED_PHONE]", out)
    return out
