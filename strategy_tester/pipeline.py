"""Pipeline class — chains stages with routing validation."""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from strategy_tester.config import DEFAULT_CONFIG
from strategy_tester.interfaces import PipelineResult, validate_interface
from strategy_tester.registry import get_method


class Pipeline:
    """Composable backtesting pipeline.

    Usage:
        pipe = Pipeline(
            s1="chan_halflife",
            s2_signal="zscore_robust_mad",
            s2_optim="grid_search",
            s3="chan_is_oos",
            s4="t_test",
            s5="equal_weight",
            name="my pipeline",
        )
        result = pipe.run(prices, pairs, config)
    """

    def __init__(
        self,
        s1: str | list[str] = "chan_halflife",
        s2_signal: str = "zscore_robust_mad",
        s2_optim: str = "grid_search",
        s3: str | list[str] = "chan_is_oos",
        s4: str | list[str] = "t_test",
        s5: str | list[str] = "equal_weight",
        name: str = "",
    ) -> None:
        self.name = name
        self.s1_name = s1
        self.s2_signal = s2_signal
        self.s2_optim = s2_optim

        def _as_list(x: str | list[str] | None) -> list[str]:
            # None means "skip this stage" (e.g. an S1+S2-only design).
            if x is None:
                return []
            return [x] if isinstance(x, str) else list(x)

        self._s1_methods = _as_list(s1)
        self._s3_methods = _as_list(s3)
        self._s4_methods = _as_list(s4)
        self._s5_methods = _as_list(s5)

        self._validate_routing()

    def _validate_routing(self) -> None:
        """Enforce S3->S4 routing constraints."""
        wfa_methods = {"wfa_expanding", "wfa_rolling"}

        if "wfe" in self._s4_methods and not (
            set(self._s3_methods) & wfa_methods
        ):
            raise ValueError(
                "WFE requires WFA output (CPCV has no IS/OOS "
                "in Pardo sense). Add wfa_expanding or "
                "wfa_rolling to S3, or remove wfe from S4."
            )

        if self.s2_signal == "bollinger" and "cpcv" in self._s3_methods:
            warnings.warn(
                "bollinger + cpcv may produce too few trades "
                "per fold for some pairs. Consider using "
                "wfa_rolling instead.",
                UserWarning,
                stacklevel=2,
            )

    def run(
        self,
        prices: pd.DataFrame,
        pairs: list[dict],
        config: dict | None = None,
        stop_after: str | None = None,
        report: bool = True,
        output_dir: Path | None = None,
    ) -> PipelineResult:
        """Execute pipeline stages sequentially."""
        t0 = time.time()
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        result = PipelineResult(name=self.name, config=cfg)

        # S1
        s1_out = self._run_s1(prices, pairs, cfg)
        result.stages["s1"] = {"result": s1_out}
        if stop_after == "s1":
            result.duration_seconds = time.time() - t0
            return result

        # S2
        s2_out = self._run_s2(prices, s1_out, cfg)
        result.stages["s2"] = {"result": s2_out}
        if stop_after == "s2":
            result.duration_seconds = time.time() - t0
            return result

        # S3
        s3_out = self._run_s3(prices, s2_out, cfg)
        result.stages["s3"] = {"result": s3_out}
        if stop_after == "s3":
            result.duration_seconds = time.time() - t0
            return result

        # S4
        s4_out = self._run_s4(s3_out, cfg)
        result.stages["s4"] = {"result": s4_out}
        if stop_after == "s4":
            result.duration_seconds = time.time() - t0
            return result

        # S5
        s5_out = self._run_s5(prices, s4_out, cfg)
        result.final = s5_out
        result.duration_seconds = time.time() - t0
        return result

    def _run_s1(
        self,
        prices: pd.DataFrame,
        pairs: list[dict],
        config: dict,
    ) -> pd.DataFrame:
        s1_name = self._s1_methods[0]
        fn = get_method("s1", s1_name)
        out = fn(prices, pairs, **config)
        validate_interface(out, "s1")
        return out

    def _run_s2(
        self,
        prices: pd.DataFrame,
        s1_result: pd.DataFrame,
        config: dict,
    ) -> pd.DataFrame:
        signal_fn = get_method("s2_signal", self.s2_signal)
        optim_fn = get_method("s2_optimize", self.s2_optim)
        out = optim_fn(prices, s1_result, signal_fn=signal_fn, **config)
        validate_interface(out, "s2")
        return out

    def _run_s3(
        self,
        prices: pd.DataFrame,
        s2_result: pd.DataFrame,
        config: dict,
    ) -> pd.DataFrame:
        signal_fn = get_method("s2_signal", self.s2_signal)
        all_results = []
        for s3_name in self._s3_methods:
            fn = get_method("s3", s3_name)
            out = fn(prices, s2_result, signal_fn=signal_fn, **config)
            all_results.append(out)

        combined = pd.concat(all_results, ignore_index=True)
        if not combined.empty:
            validate_interface(combined, "s3")
        return combined

    def _run_s4(
        self,
        s3_result: pd.DataFrame,
        config: dict,
    ) -> pd.DataFrame:
        all_results = []
        for s4_name in self._s4_methods:
            fn = get_method("s4", s4_name)
            out = fn(s3_result, **config)
            all_results.append(out)

        if not all_results:
            return pd.DataFrame(
                columns=["pair", "numerator", "denominator", "passed", "tier"]
            )

        combined = pd.concat(all_results, ignore_index=True)
        if not combined.empty:
            validate_interface(combined, "s4")

            tier_order = {"TOP_TIER": 0, "SECOND_TIER": 1, "REJECT": 2}
            combined["_tier_ord"] = combined["tier"].map(
                lambda t: tier_order.get(t, 2)
            )
            combined = (
                combined.sort_values("_tier_ord")
                .drop_duplicates(subset=["pair"], keep="first")
                .drop(columns=["_tier_ord"])
            )
        return combined

    def _run_s5(
        self,
        prices: pd.DataFrame,
        s4_result: pd.DataFrame,
        config: dict,
    ) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for s5_name in self._s5_methods:
            fn = get_method("s5", s5_name)
            results[s5_name] = fn(prices, s4_result, **config)
        return results
