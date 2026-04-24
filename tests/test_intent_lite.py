"""Tests for the keyword-based intent tagger (no external deps)."""

from pipeline.intent_lite import tag_intent


def test_billing_intent():
    assert tag_intent("I want a refund on my subscription") == "billing"


def test_account_intent():
    assert tag_intent("How do I reset my password?") == "account"


def test_shipping_intent():
    assert tag_intent("Where is my order? The package hasn't arrived") == "shipping"


def test_technical_intent():
    assert tag_intent("The app keeps crashing with an error") == "technical"


def test_general_fallback():
    # No matching keywords — should fall back to general
    assert tag_intent("Lorem ipsum dolor sit amet") == "general"


def test_empty_query():
    assert tag_intent("") == "general"
