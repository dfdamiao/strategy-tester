"""Thin argparse wrapper around lib.pipeline.

Surface mirrors `lib/docs/CLI_REFERENCE.md`. Knobs only — methods stay in
config.yaml (CLI_REFERENCE.md §2). Overrides flow through to `run()` as a
flat override dict; lib.io builds the human-readable tag suffix per §11.

Status: Phase 1.E foundation (2026-05-17). Pipeline.run integration with
hooks/anomaly comes after the parity audit (Phase 1.C).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from strategy_tester.anomaly import run_checks
from strategy_tester.hooks import PipelineContext, fire
from strategy_tester.io import build_tag

LOGGER = logging.getLogger("strategy_tester.cli")


# Knob keys whose presence in the merged-CLI dict triggers tag suffixing.
# Order is preserved by sorted() in build_tag().
_KNOB_KEYS: tuple[str, ...] = (
    "s0_min_nonnan",
    "s1_min_pass_rate",
    "s1_max_pass_rate",
    "s2_is_ratio",
    "s2_grid_density",
    "s2_min_trades_gate",
    "s2_costs_bps",
    "s2_oos_sr_max",
    "s2_mode",
    "s3_folds",
    "s3_cpcv_k",
    "s3_cpcv_n",
    "s3_pooled_sr_min",
    "s3_fold_win_min",
    "s3_gate_mode",
    "s3_pbo_max",
    "s3_purge_pct",
    "s3_embargo_pct",
    "s4_psr",
    "s4_dsr",
    "s4_use_effective_n",
    "s4_rho_bar",
    "s4_min_trades",
    "s4_bootstrap_n",
    "s4_bootstrap_conf",
    "s4_pass_rate_min",
    "s4_pass_rate_max",
    "s4_cohort",
    "s5_cohorts",
    "s5_benchmark",
    "s5_cash_buffer",
    "s5_seed_nav",
    "s5_roll_window",
    "s5_cost_bps",
    "s5_haircut",
    "s5_selection_metric",
    "s5_ir_min",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the canonical argparse parser per CLI_REFERENCE.md."""
    ap = argparse.ArgumentParser(
        prog="python -m strategy_tester.pipeline",
        description=(
            "Run a strategy pipeline from config.yaml. Knob overrides "
            "via CLI; method changes via yaml only. See "
            "strategy_tester/docs/CLI_REFERENCE.md."
        ),
    )
    sub = ap.add_subparsers(dest="command", required=True)
    run_ap = sub.add_parser("run", help="Execute pipeline")
    _add_run_args(run_ap)
    sub.add_parser("list-methods", help="List registered methods")
    sub.add_parser("list-checks", help="List registered anomaly checks")
    sub.add_parser("list-hooks", help="List registered hooks")
    return ap


def _add_run_args(ap: argparse.ArgumentParser) -> None:
    """Attach the full flag surface to the `run` subcommand."""
    # --- Required ---
    ap.add_argument("--config", required=True, type=Path,
                    help="Path to validation/config.yaml")

    # --- Universal ---
    ap.add_argument("--stop-after", choices=["s0", "s1", "s2", "s3", "s4", "s5"],
                    default="s5")
    ap.add_argument("--start-from",
                    choices=["auto", "s0", "s1", "s2", "s3", "s4", "s5"],
                    default="auto")
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--resume", dest="resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--fresh", action="store_true",
                    help="Wipe cache/ before run")
    ap.add_argument("--anomaly-policy",
                    choices=["off", "warn", "strict"], default="warn")
    ap.add_argument("--strict-on", default="",
                    help="Comma-separated check names to escalate to strict")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", type=int, choices=[0, 1, 2, 3], default=1)
    ap.add_argument("--log-file", type=Path, default=None)
    ap.add_argument("--output-suffix", default="",
                    help="Manual tag override; bypasses auto-derived tag")

    # --- S0 ---
    ap.add_argument("--s0-skip", action="store_true")
    ap.add_argument("--s0-min-nonnan", type=float, default=None)

    # --- S1 ---
    ap.add_argument("--s1-min-pass-rate", type=float, default=None)
    ap.add_argument("--s1-max-pass-rate", type=float, default=None)
    ap.add_argument("--s1-halflife-min", type=int, default=None)
    ap.add_argument("--s1-halflife-max", type=int, default=None)
    ap.add_argument("--s1-hurst-max", type=float, default=None)
    ap.add_argument("--s1-hurst-min", type=float, default=None)
    ap.add_argument("--s1-er-min", type=float, default=None)
    ap.add_argument("--s1-er-max", type=float, default=None)

    # --- S2 ---
    ap.add_argument("--s2-is-ratio", type=float, default=None)
    ap.add_argument("--s2-grid-density",
                    choices=["coarse", "full"], default=None)
    ap.add_argument("--s2-min-trades-gate", type=int, default=None)
    ap.add_argument("--s2-costs-bps", type=float, default=None)
    ap.add_argument("--s2-oos-sr-max", type=float, default=None)
    ap.add_argument("--s2-mode",
                    choices=["per_asset", "no_sweep"], default=None)

    # --- S3 ---
    ap.add_argument("--s3-folds", type=int, default=None)
    ap.add_argument("--s3-cpcv-k", type=int, default=None)
    ap.add_argument("--s3-cpcv-n", type=int, default=None)
    ap.add_argument("--s3-pooled-sr-min", type=float, default=None)
    ap.add_argument("--s3-fold-win-min", type=float, default=None)
    ap.add_argument("--s3-gate-mode",
                    choices=["1_pooled", "1_winrate", "2_and", "3_and_pbo"],
                    default=None)
    ap.add_argument("--s3-pbo-max", type=float, default=None)
    ap.add_argument("--s3-purge-pct", type=float, default=None)
    ap.add_argument("--s3-embargo-pct", type=float, default=None)
    ap.add_argument("--reuse-s3", action="store_true")

    # --- S4 ---
    ap.add_argument("--s4-psr", type=float, default=None)
    ap.add_argument("--s4-dsr", type=float, default=None)
    ap.add_argument("--s4-use-effective-n", dest="s4_use_effective_n",
                    action="store_true", default=None)
    ap.add_argument("--no-s4-use-effective-n", dest="s4_use_effective_n",
                    action="store_false")
    ap.add_argument("--s4-rho-bar", type=float, default=None)
    ap.add_argument("--s4-min-trades", type=int, default=None)
    ap.add_argument("--s4-bootstrap-n", type=int, default=None)
    ap.add_argument("--s4-bootstrap-conf", type=float, default=None)
    ap.add_argument("--s4-pass-rate-min", type=float, default=None)
    ap.add_argument("--s4-pass-rate-max", type=float, default=None)
    ap.add_argument(
        "--s4-cohort",
        choices=["passed_s4", "passed_s4_strict", "psr_pass",
                 "psr_pass_strict", "dsr_pass", "dsr_pass_strict"],
        default=None,
    )

    # --- S5 ---
    ap.add_argument("--s5-cohorts", default=None,
                    help="Comma-separated subset of {singles, ratios, combined}")
    ap.add_argument("--s5-benchmark", default=None)
    ap.add_argument("--s5-cash-buffer", type=float, default=None)
    ap.add_argument("--s5-seed-nav", type=float, default=None)
    ap.add_argument("--s5-roll-window", type=int, default=None)
    ap.add_argument("--s5-cost-bps", type=float, default=None)
    ap.add_argument("--s5-haircut", dest="s5_haircut",
                    action="store_true", default=None)
    ap.add_argument("--no-s5-haircut", dest="s5_haircut", action="store_false")
    ap.add_argument("--s5-selection-metric",
                    choices=["ir_vs_spy", "oos_calmar", "oos_sr_full", "alpha"],
                    default=None)
    ap.add_argument("--s5-ir-min", type=float, default=None)
    ap.add_argument("--no-html", action="store_true")
    ap.add_argument("--html-only", action="store_true")
    ap.add_argument("--export-parquet", dest="export_parquet",
                    action="store_true", default=None)
    ap.add_argument("--no-export-parquet", dest="export_parquet",
                    action="store_false")


def parse_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Extract the override knobs from parsed args (drops None values)."""
    overrides: dict[str, Any] = {}
    for key in _KNOB_KEYS:
        val = getattr(args, key, None)
        if val is not None:
            overrides[key] = val
    return overrides


def derive_tag(args: argparse.Namespace, overrides: dict[str, Any]) -> str:
    """Compute the artifact-suffix tag.

    Honour `--output-suffix` if set; otherwise auto-derive from overrides.
    """
    if args.output_suffix:
        return args.output_suffix
    return build_tag(overrides)


def load_config(path: Path) -> dict[str, Any]:
    """Load config.yaml (lazy yaml import — pydantic config still wins)."""
    import yaml

    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def merge_config(
    base: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Merge CLI knob overrides into the yaml config (deep merge by path)."""
    merged = json.loads(json.dumps(base))  # cheap deep copy
    for flat_key, val in overrides.items():
        path = _flat_to_path(flat_key)
        _set_path(merged, path, val)
    return merged


def _flat_to_path(flat_key: str) -> list[str]:
    """Map flat flag name → yaml path.

    Examples
    --------
    >>> _flat_to_path("s4_psr")
    ['stages', 's4', 'psr_threshold']
    >>> _flat_to_path("s3_folds")
    ['stages', 's3', 'n_folds']
    """
    mapping = {
        "s0_min_nonnan": ["stages", "s0", "min_nonnan"],
        "s1_min_pass_rate": ["stages", "s1", "min_pass_rate"],
        "s1_max_pass_rate": ["stages", "s1", "max_pass_rate"],
        "s2_is_ratio": ["stages", "s2", "is_ratio"],
        "s2_grid_density": ["stages", "s2", "grid_density"],
        "s2_min_trades_gate": ["stages", "s2", "min_trades_gate"],
        "s2_costs_bps": ["stages", "s2", "costs_bps"],
        "s2_oos_sr_max": ["stages", "s2", "oos_sr_max"],
        "s2_mode": ["stages", "s2", "mode"],
        "s3_folds": ["stages", "s3", "n_folds"],
        "s3_cpcv_k": ["stages", "s3", "cpcv", "k"],
        "s3_cpcv_n": ["stages", "s3", "cpcv", "n_test"],
        "s3_pooled_sr_min": [
            "stages", "s3", "pass_gate", "pooled_oos_sharpe_min",
        ],
        "s3_fold_win_min": [
            "stages", "s3", "pass_gate", "fold_win_rate_min",
        ],
        "s3_gate_mode": ["stages", "s3", "pass_gate", "mode"],
        "s3_pbo_max": ["stages", "s3", "pass_gate", "pbo_max"],
        "s3_purge_pct": ["stages", "s3", "cpcv", "purge_pct"],
        "s3_embargo_pct": ["stages", "s3", "cpcv", "embargo_pct"],
        "s4_psr": ["stages", "s4", "psr_threshold"],
        "s4_dsr": ["stages", "s4", "dsr_threshold"],
        "s4_use_effective_n": ["stages", "s4", "use_effective_n"],
        "s4_rho_bar": ["stages", "s4", "rho_bar"],
        "s4_min_trades": ["stages", "s4", "min_trades_full"],
        "s4_bootstrap_n": ["stages", "s4", "bootstrap_n"],
        "s4_bootstrap_conf": ["stages", "s4", "bootstrap_conf"],
        "s4_pass_rate_min": ["stages", "s4", "pass_rate_min"],
        "s4_pass_rate_max": ["stages", "s4", "pass_rate_max"],
        "s4_cohort": ["stages", "s4", "cohort_select"],
        "s5_cohorts": ["stages", "s5", "cohorts"],
        "s5_benchmark": ["stages", "s5", "benchmark"],
        "s5_cash_buffer": ["stages", "s5", "cash_buffer"],
        "s5_seed_nav": ["stages", "s5", "seed_nav"],
        "s5_roll_window": ["stages", "s5", "roll_window"],
        "s5_cost_bps": ["stages", "s5", "cost_bps"],
        "s5_haircut": ["stages", "s5", "apply_dsr_haircut"],
        "s5_selection_metric": ["stages", "s5", "selection_metric"],
        "s5_ir_min": ["stages", "s5", "ir_min"],
    }
    return mapping.get(flat_key, [flat_key])


def _set_path(d: dict[str, Any], path: list[str], val: Any) -> None:
    cur = d
    for k in path[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[path[-1]] = val


def _write_effective_config(
    merged: dict[str, Any],
    out_dir: Path,
    tag: str,
) -> Path:
    """Write the merged config + run timestamp for reproducibility."""
    import yaml

    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    name = f"run_{timestamp}{'_' + tag if tag else ''}.yaml"
    path = out_dir / name
    with path.open("w") as fh:
        yaml.safe_dump(
            {"meta": {"timestamp": timestamp, "tag": tag},
             "config": merged},
            fh,
            sort_keys=False,
        )
    return path


def main(argv: list[str] | None = None) -> int:
    """Entry point — returns Unix exit code."""
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.command == "list-methods":
        from strategy_tester.registry import list_methods

        print(json.dumps(list_methods(), indent=2))
        return 0

    if args.command == "list-checks":
        from strategy_tester.anomaly import list_checks

        print(json.dumps(list_checks(), indent=2))
        return 0

    if args.command == "list-hooks":
        from strategy_tester.hooks import list_hooks

        print(json.dumps(list_hooks(), indent=2))
        return 0

    # run command
    logging.basicConfig(
        level={0: logging.ERROR, 1: logging.INFO,
               2: logging.DEBUG, 3: logging.DEBUG}[args.verbose],
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.config.exists():
        LOGGER.error(f"Config not found: {args.config}")
        return 1

    base_config = load_config(args.config)
    overrides = parse_overrides(args)
    merged = merge_config(base_config, overrides)
    tag = derive_tag(args, overrides)

    # Resolve output dir: config has it, or derive from config path.
    # `parent.parent` only works for the canonical
    # `<strategy>/validation/config.yaml` layout — fall back to sibling
    # `results/` directory otherwise.
    if merged.get("output_dir"):
        out_dir = Path(merged["output_dir"])
    else:
        candidate = args.config.resolve().parent.parent / "results"
        if not candidate.parent.exists():
            candidate = args.config.resolve().parent / "results"
        out_dir = candidate

    # Always write the effective config — even on --dry-run
    effective_path = _write_effective_config(merged, out_dir, tag)
    LOGGER.info(f"Effective config: {effective_path}")

    ctx = PipelineContext(
        strategy=merged.get("strategy", "unknown"),
        config=merged,
        output_dir=out_dir,
        logger=LOGGER,
        run_id=tag or "canonical",
    )

    if args.dry_run:
        LOGGER.info("Dry run — config + tag emitted; no compute.")
        fire("pre_pipeline", ctx=ctx)
        fire("post_pipeline", ctx=ctx, final_outputs={})
        return 0

    # Phase 1.E stub — the actual pipeline.run integration lives here.
    # Today the existing Pipeline class still drives execution from
    # per-strategy pipeline.py scripts. This CLI is the future entry point.
    LOGGER.warning(
        "lib.cli.run dispatch is a Phase 1.E stub — "
        "today's per-strategy pipeline.py scripts still drive execution. "
        "Use this CLI to validate config + emit effective.yaml + dry-run."
    )

    # Demo: fire pre/post-pipeline hooks (anomaly + run_log writer wired)
    fire("pre_pipeline", ctx=ctx)
    fire("post_pipeline", ctx=ctx, final_outputs={})

    # Demonstrate anomaly + tag-output convention by running checks on
    # any pre-existing stage CSVs found in out_dir (read-only inspection).
    strict_on = {s.strip() for s in args.strict_on.split(",") if s.strip()}
    n_strict = 0
    for stage in ("s1", "s2", "s3", "s4", "s5"):
        if args.stop_after and stage > args.stop_after:
            break
        # Look for a stage CSV in out_dir
        candidates = list(out_dir.glob(f"{stage}*metrics*.csv"))
        if not candidates:
            continue
        try:
            import pandas as pd

            df = pd.read_csv(candidates[0])
        except OSError:
            continue
        anomalies = run_checks(
            stage,
            df,
            policy=args.anomaly_policy,
            strict_on=strict_on,
        )
        for a in anomalies:
            fire("on_anomaly", ctx=ctx, anomaly=a)
            if a.severity == "strict":
                n_strict += 1

    if n_strict and args.anomaly_policy != "off":
        LOGGER.error(f"{n_strict} strict-severity anomaly(ies) — exit 2")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
