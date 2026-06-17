"""Tests for the method registry."""
from __future__ import annotations
import pytest
from strategy_tester.registry import (
    register_stage, get_method, list_methods,
)


def test_register_and_retrieve():
    @register_stage("s1")
    def _test_method(prices, pairs, **config):
        return "ok"

    fn = get_method("s1", "_test_method")
    assert fn is _test_method
    assert fn(None, None) == "ok"


def test_get_unknown_raises():
    with pytest.raises(KeyError, match="no_such_method"):
        get_method("s1", "no_such_method")


def test_list_methods_all():
    result = list_methods()
    assert isinstance(result, dict)
    assert "s1" in result


def test_list_methods_single_stage():
    result = list_methods("s1")
    assert isinstance(result, dict)
    assert len(result) == 1


def test_invalid_stage_raises():
    with pytest.raises(ValueError, match="Unknown stage"):
        register_stage("s99")(lambda: None)
