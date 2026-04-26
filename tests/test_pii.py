"""Tests for `api.pii.redact_pii`."""

from __future__ import annotations

from api.pii import redact_pii


def test_email_redacted():
    out = redact_pii("Contact me at john.doe+test@example.com please.")
    assert "john.doe+test@example.com" not in out
    assert "[REDACTED_EMAIL]" in out


def test_phone_us_format_redacted():
    out = redact_pii("Call (415) 555-0199 anytime.")
    assert "555-0199" not in out
    assert "[REDACTED_PHONE]" in out


def test_phone_international_redacted():
    out = redact_pii("My number is +1 415 555 0199.")
    assert "555 0199" not in out
    assert "[REDACTED_PHONE]" in out


def test_credit_card_redacted():
    out = redact_pii("Card: 4111 1111 1111 1111 expires 12/26")
    assert "4111 1111 1111 1111" not in out
    assert "[REDACTED_CC]" in out


def test_ssn_redacted():
    out = redact_pii("SSN 123-45-6789")
    assert "123-45-6789" not in out
    assert "[REDACTED_SSN]" in out


def test_clean_input_unchanged():
    text = "How do I reset my password?"
    assert redact_pii(text) == text


def test_empty_input():
    assert redact_pii("") == ""
    assert redact_pii(None) == ""


def test_multiple_pii_types_in_one_string():
    out = redact_pii("Email me at a@b.co or call 555-867-5309. SSN 111-22-3333.")
    assert "a@b.co" not in out
    assert "555-867-5309" not in out
    assert "111-22-3333" not in out
    assert "[REDACTED_EMAIL]" in out
    assert "[REDACTED_PHONE]" in out
    assert "[REDACTED_SSN]" in out
