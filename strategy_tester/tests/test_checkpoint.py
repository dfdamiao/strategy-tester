"""Tests for checkpoint/resume system."""
from __future__ import annotations
from pathlib import Path
from strategy_tester.checkpoint import StageCheckpoint


def test_checkpoint_write_and_resume(tmp_path: Path):
    ckpt = StageCheckpoint("s1", tmp_path)
    ckpt.save_pair_result("A/B", {"pair": "A/B", "passed": True, "halflife": 20})
    ckpt.save_pair_result("C/D", {"pair": "C/D", "passed": False, "halflife": 500})

    # Resume: new checkpoint loads existing
    ckpt2 = StageCheckpoint("s1", tmp_path)
    completed = ckpt2.load_completed()
    assert "A/B" in completed
    assert "C/D" in completed
    assert len(completed) == 2


def test_checkpoint_final_output(tmp_path: Path):
    ckpt = StageCheckpoint("s1", tmp_path)
    ckpt.save_pair_result("A/B", {"pair": "A/B", "passed": True})
    final = ckpt.finalize()
    assert final.exists()
    assert final.name == "s1_results_latest.csv"


def test_checkpoint_cleanup(tmp_path: Path):
    ckpt = StageCheckpoint("s1", tmp_path)
    ckpt.save_pair_result("A/B", {"pair": "A/B", "passed": True})
    assert ckpt.csv_path.exists()
    ckpt.finalize()
    ckpt.cleanup()
    assert not ckpt.csv_path.exists()


def test_checkpoint_engine_data(tmp_path: Path):
    ckpt = StageCheckpoint("s1", tmp_path)
    ckpt.save_pair_result(
        "A/B", {"pair": "A/B", "passed": True},
        engine_data={"returns": [0.01, -0.02, 0.03]},
    )
    # Resume
    ckpt2 = StageCheckpoint("s1", tmp_path)
    assert "A/B" in ckpt2._engine_data
    assert ckpt2._engine_data["A/B"]["returns"] == [0.01, -0.02, 0.03]
