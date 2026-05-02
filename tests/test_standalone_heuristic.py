"""Tests for the cheap standalone-question heuristic in LangChainRAG.

The heuristic decides whether we can skip the condense-chain LLM call.
It needs to be conservative: any false positive (treating a follow-up as
standalone) hurts retrieval quality. False negatives only cost extra latency.
"""

from pipeline.rag import LangChainRAG


def test_short_factual_query_is_standalone():
    assert LangChainRAG._looks_standalone("How do I reset my password?")
    assert LangChainRAG._looks_standalone("Where is my order?")
    assert LangChainRAG._looks_standalone("How do I get a refund?")


def test_pronoun_followups_need_condensing():
    assert not LangChainRAG._looks_standalone("Can I do it without an email?")
    assert not LangChainRAG._looks_standalone("What about them?")
    assert not LangChainRAG._looks_standalone("Tell me more about this.")
    assert not LangChainRAG._looks_standalone("How does she change it?")


def test_reference_phrases_need_condensing():
    assert not LangChainRAG._looks_standalone("The same for shipping?")
    assert not LangChainRAG._looks_standalone("Like before, please.")
    assert not LangChainRAG._looks_standalone("Can you do that again?")


def test_long_queries_treated_as_standalone():
    long_q = (
        "I tried to reset my password three times today using the forgot "
        "password link but I never received the confirmation email even "
        "though I checked my spam folder carefully — what should I do next?"
    )
    assert LangChainRAG._looks_standalone(long_q)


def test_empty_query_is_standalone():
    # Empty input shouldn't trigger an LLM call.
    assert LangChainRAG._looks_standalone("")
    assert LangChainRAG._looks_standalone("   ")
