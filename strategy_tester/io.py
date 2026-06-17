"""Schema-validated I/O for stage artifacts.

Wraps parquet / csv read+write with pydantic validation from ``lib.schemas``.
Centralises the file naming convention from CLI_REFERENCE.md §11 (human-
readable tag suffix per override set).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from strategy_tester.schemas import validate_dataframe

DEFAULT_FORMAT = "parquet"


def write_stage_output(
    df: pd.DataFrame,
    stage: str,
    out_dir: Path,
    *,
    cohort: str | None = None,
    tag: str = "",
    fmt: str = DEFAULT_FORMAT,
    validate: bool = True,
) -> Path:
    """Write a stage DataFrame with schema validation + tag suffix.

    File naming: ``{stage}_{cohort}_{tag}_metrics.{ext}``
        - cohort omitted when None (used for S1-S4)
        - tag omitted when empty (canonical run)

    Examples
    --------
    >>> write_stage_output(df, "s4", out_dir)
    # → out_dir / "s4_metrics.parquet"
    >>> write_stage_output(df, "s5", out_dir, cohort="ratios", tag="psr95")
    # → out_dir / "s5_ratios_psr95_metrics.parquet"
    """
    if validate:
        validate_dataframe(df, stage)

    out_dir.mkdir(parents=True, exist_ok=True)
    name = _build_filename(stage, cohort=cohort, tag=tag, fmt=fmt)
    path = out_dir / name

    if fmt == "parquet":
        df.to_parquet(path, index=False)
    elif fmt == "csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported format {fmt!r} (expected parquet|csv)")

    return path


def read_stage_output(
    path: Path,
    stage: str,
    *,
    validate: bool = True,
) -> pd.DataFrame:
    """Read a stage artifact and (optionally) validate against the schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(
            f"Unsupported extension {path.suffix!r} (expected .parquet|.csv)"
        )

    if validate:
        validate_dataframe(df, stage)
    return df


def _build_filename(
    stage: str,
    *,
    cohort: str | None,
    tag: str,
    fmt: str,
) -> str:
    """Build canonical filename per CLI_REFERENCE.md §11."""
    parts = [stage]
    if cohort:
        parts.append(cohort)
    if tag:
        parts.append(tag)
    parts.append("metrics")
    ext = "parquet" if fmt == "parquet" else "csv"
    return f"{'_'.join(parts)}.{ext}"


def build_tag(overrides: dict[str, object], *, max_length: int = 40) -> str:
    """Build a human-readable tag from CLI override dict.

    Maps e.g. {"s4_psr": 0.95, "s4_dsr": 0.90} → "psr95_dsr90".
    Falls back to short hash if joined tag exceeds ``max_length``.

    See CLI_REFERENCE.md §11 for the contract.
    """
    if not overrides:
        return ""

    # Canonical key shortenings (covers the common knobs)
    short = {
        "s4_psr": "psr",
        "s4_dsr": "dsr",
        "s3_folds": "s3folds",
        "s3_cpcv_k": "cpcvK",
        "s3_cpcv_n": "cpcvN",
        "s2_grid_density": "grid",
        "s2_is_ratio": "is",
        "s5_cash_buffer": "cash",
        "s5_roll_window": "roll",
        "s4_cohort": "cohort",
    }

    parts: list[str] = []
    for key in sorted(overrides):
        val = overrides[key]
        short_key = short.get(key, key.replace("_", ""))
        if isinstance(val, float):
            val_str = f"{int(round(val * 100))}"
        elif isinstance(val, bool):
            val_str = "on" if val else "off"
        else:
            val_str = str(val).replace("_", "")
        parts.append(f"{short_key}{val_str}")

    tag = "_".join(parts)
    if len(tag) > max_length:
        import hashlib

        digest = hashlib.sha256(tag.encode()).hexdigest()[:8]
        return digest
    return tag
