import pytest

from app.llm import _extract_json


def test_plain_json_object():
    assert _extract_json('{"a": 1, "b": "two"}') == {"a": 1, "b": "two"}


def test_json_wrapped_in_code_fence():
    text = '```json\n{"a": 1}\n```'
    assert _extract_json(text) == {"a": 1}


def test_json_wrapped_in_bare_code_fence():
    text = '```\n{"a": 1}\n```'
    assert _extract_json(text) == {"a": 1}


def test_json_with_surrounding_prose():
    text = 'Here is the result:\n{"a": 1}\nHope that helps!'
    assert _extract_json(text) == {"a": 1}


def test_no_json_raises_value_error():
    with pytest.raises(ValueError):
        _extract_json("no json here at all")
