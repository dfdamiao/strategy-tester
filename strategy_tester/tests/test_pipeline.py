"""Tests for Pipeline class."""
from __future__ import annotations

import pytest

# Trigger all registrations
import strategy_tester  # noqa: F401
from strategy_tester.pipeline import Pipeline


def test_pipeline_init_valid() -> None:
    pipe = Pipeline(
        s1="chan_halflife",
        s2_signal="zscore_robust_mad",
        s2_optim="grid_search",
        s3="chan_is_oos",
        s4="t_test",
        s5="equal_weight",
        name="test",
    )
    assert pipe.name == "test"


def test_pipeline_routing_cpcv_wfe_raises() -> None:
    with pytest.raises(ValueError, match="WFE requires WFA"):
        Pipeline(
            s1="chan_halflife",
            s2_signal="zscore_robust_mad",
            s2_optim="grid_search",
            s3="cpcv",
            s4="wfe",
            s5="equal_weight",
        )


def test_pipeline_routing_bollinger_cpcv_warns() -> None:
    with pytest.warns(UserWarning, match="bollinger.*cpcv"):
        Pipeline(
            s1="chan_halflife",
            s2_signal="bollinger",
            s2_optim="grid_search",
            s3="cpcv",
            s4="dsr",
            s5="equal_weight",
        )


def test_pipeline_multi_s3() -> None:
    pipe = Pipeline(
        s1="chan_combined",
        s2_signal="zscore_robust_mad",
        s2_optim="grid_search",
        s3=["wfa_expanding", "cpcv"],
        s4=["dsr", "wfe"],
        s5="handcraft_carver",
        name="X-11",
    )
    assert len(pipe._s3_methods) == 2
