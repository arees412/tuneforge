from __future__ import annotations

from pathlib import Path

import pytest

from tuneforge.scenarios import run_scenarios


@pytest.mark.training
def test_all_twelve_acceptance_scenarios(tmp_path: Path) -> None:
    result = run_scenarios(tmp_path / "acceptance")
    assert result["all_passed"]
    assert len([key for key in result if key[:2].isdigit()]) == 12
    assert (
        result["06_tiny_lora_training"]["parameters"]["trainable_parameters"]
        < result["06_tiny_lora_training"]["parameters"]["total_parameters"]
    )
    assert not result["07_unsupported_qlora"]["executed"]
    assert result["12_non_finite_loss"]["checkpoint_count"] == 0
