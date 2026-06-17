"""Checkpoint/resume for long-running pipeline stages."""
from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Any


class StageCheckpoint:
    """Checkpoint/resume for long-running stages.

    After each pair, appends to CSV + saves engine state.
    On resume, loads completed pairs and skips them.
    """

    def __init__(self, stage: str, output_dir: Path) -> None:
        self.stage = stage
        self.output_dir = output_dir
        self.csv_path = output_dir / f"{stage}_checkpoint.csv"
        self.pkl_path = output_dir / f"{stage}_checkpoint.pkl"
        self._completed: set[str] = set()
        self._rows: list[dict] = []
        self._engine_data: dict[str, Any] = {}

        # Load existing checkpoint if resuming
        if self.csv_path.exists():
            self._load_existing()

    def _load_existing(self) -> None:
        """Load completed pairs from existing checkpoint."""
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._completed.add(row.get("pair", ""))
                self._rows.append(row)
        if self.pkl_path.exists():
            with open(self.pkl_path, "rb") as f:
                self._engine_data = pickle.load(f)

    def load_completed(self) -> set[str]:
        """Return set of pair names already completed."""
        return set(self._completed)

    def save_pair_result(
        self,
        pair: str,
        row: dict,
        engine_data: Any = None,
    ) -> None:
        """Append one pair result to checkpoint files."""
        self._completed.add(pair)
        self._rows.append(row)

        # Append to CSV
        file_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        # Save engine data
        if engine_data is not None:
            self._engine_data[pair] = engine_data
            with open(self.pkl_path, "wb") as f:
                pickle.dump(self._engine_data, f)

    def finalize(self) -> Path:
        """Write final _latest.csv from checkpoint data.
        Returns path to final output file."""
        final_path = self.output_dir / f"{self.stage}_results_latest.csv"
        if self._rows:
            with open(final_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=self._rows[0].keys(),
                )
                writer.writeheader()
                writer.writerows(self._rows)
        else:
            final_path.write_text("")
        return final_path

    def cleanup(self) -> None:
        """Remove checkpoint files after successful finalize."""
        if self.csv_path.exists():
            self.csv_path.unlink()
        if self.pkl_path.exists():
            self.pkl_path.unlink()
