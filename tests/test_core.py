"""Tests for core utilities: IdGenerator, Clock, Hasher, Redactor, RefSelector, TokenEstimator."""

from __future__ import annotations

import pytest

from context_engineering_sdk.core.clock import SystemClock
from context_engineering_sdk.core.errors import SelectorResolveError
from context_engineering_sdk.core.hasher import Sha256Hasher
from context_engineering_sdk.core.id_generator import UuidV4Generator
from context_engineering_sdk.core.redactor import RegexRedactor, RedactionRules
from context_engineering_sdk.core.ref_selector import RefSelector
from context_engineering_sdk.core.token_estimator import CharBasedEstimator
from context_engineering_sdk.core.types import Message, Role


# ---- IdGenerator ----

def test_uuid_generator_uniqueness():
    gen = UuidV4Generator()
    ids = {gen.generate() for _ in range(100)}
    assert len(ids) == 100


def test_uuid_generator_format():
    gen = UuidV4Generator()
    uid = gen.generate()
    assert len(uid) == 36
    assert uid.count("-") == 4


# ---- Clock ----

def test_system_clock_iso_format():
    clock = SystemClock()
    ts = clock.now_iso()
    assert "T" in ts
    assert "+" in ts or "Z" in ts or ts.endswith("+00:00")


# ---- Hasher ----

def test_sha256_deterministic():
    h = Sha256Hasher()
    assert h.digest("hello") == h.digest("hello")


def test_sha256_different_inputs():
    h = Sha256Hasher()
    assert h.digest("hello") != h.digest("world")


def test_sha256_hex_length():
    h = Sha256Hasher()
    assert len(h.digest("test")) == 64


# ---- Redactor ----

@pytest.mark.asyncio
async def test_redactor_email():
    r = RegexRedactor()
    result = await r.redact("Contact me at john@example.com")
    assert "[REDACTED:email]" in result.redacted_text
    assert result.has_sensitive
    assert "email" in result.applied_rules


@pytest.mark.asyncio
async def test_redactor_phone():
    r = RegexRedactor()
    result = await r.redact("Call me at 555-123-4567")
    assert "[REDACTED:phone_number]" in result.redacted_text


@pytest.mark.asyncio
async def test_redactor_no_sensitive():
    r = RegexRedactor()
    result = await r.redact("This is a normal text without sensitive data.")
    assert result.redacted_text == "This is a normal text without sensitive data."
    assert not result.has_sensitive


@pytest.mark.asyncio
async def test_redactor_custom_patterns():
    r = RegexRedactor()
    rules = RedactionRules(patterns=[r"SECRET-\w+"])
    result = await r.redact("The key is SECRET-ABC123", rules=rules)
    assert "[REDACTED:custom_0]" in result.redacted_text


# ---- RefSelector ----

def test_ref_selector_lines():
    sel = RefSelector()
    content = "line1\nline2\nline3\nline4\nline5"
    result = sel.extract(content, "lines:2-4")
    assert result == "line2\nline3\nline4"


def test_ref_selector_chars():
    sel = RefSelector()
    content = "Hello World, this is a test"
    result = sel.extract(content, "chars:6-11")
    assert result == "World"


def test_ref_selector_regex():
    sel = RefSelector()
    content = "def hello(): pass\ndef world(): pass"
    result = sel.extract(content, r"regex:def\s+\w+\(\)")
    assert result == "def hello()"


def test_ref_selector_json():
    sel = RefSelector()
    import json
    content = json.dumps({"data": {"items": [{"name": "foo"}, {"name": "bar"}]}})
    result = sel.extract(content, "json:$.data.items[0].name")
    assert result == "foo"


def test_ref_selector_invalid_format():
    sel = RefSelector()
    with pytest.raises(SelectorResolveError):
        sel.extract("content", "invalid")


def test_ref_selector_invalid_lines():
    sel = RefSelector()
    with pytest.raises(SelectorResolveError):
        sel.extract("content", "lines:abc")


def test_ref_selector_parse_multiple():
    sel = RefSelector()
    parsed = sel.parse("lines:1-5,chars:0-10")
    assert len(parsed) == 2
    assert parsed[0].type == "lines"
    assert parsed[1].type == "chars"


# ---- TokenEstimator ----

def test_char_estimator_english():
    est = CharBasedEstimator()
    tokens = est.estimate_text("Hello world this is a test")
    assert tokens > 0


def test_char_estimator_chinese():
    est = CharBasedEstimator()
    tokens = est.estimate_text("你好世界这是一个测试")
    assert tokens > 10  # CJK chars should estimate higher


def test_char_estimator_empty():
    est = CharBasedEstimator()
    assert est.estimate_text("") == 0


def test_char_estimator_message():
    est = CharBasedEstimator()
    tokens = est.estimate_message(Role.USER, "Hello world")
    assert tokens > est.estimate_text("Hello world")


def test_char_estimator_messages():
    est = CharBasedEstimator()
    messages = [
        Message(role=Role.USER, content="Hello"),
        Message(role=Role.ASSISTANT, content="Hi there!"),
    ]
    tokens = est.estimate_messages(messages)
    assert tokens > 0
