from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = (
    ROOT
    / "skills"
    / "materials-science"
    / "dpa4-finetuning"
    / "scripts"
    / "run_finetuning.py"
)
TEMPLATE = SCRIPT.parents[1] / "assets" / "input.json"


def _load_script():
    spec = importlib.util.spec_from_file_location("dpa4_finetuning_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dataset(root: Path, frames: dict[str, int]) -> Path:
    for name, count in frames.items():
        system = root / name
        batch = system / "set.000"
        batch.mkdir(parents=True)
        (system / "type.raw").write_text("0\n", encoding="utf-8")
        (system / "type_map.raw").write_text("Ag\n", encoding="utf-8")
        np.save(batch / "coord.npy", np.zeros((count, 3)))
    return root


def _fake_dp(path: Path, *, fail_step: str = "") -> Path:
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "step=\"$2\"\n"
        "printf '%s\\n' \"$step\" >> command-order\n"
        f"if [ \"$step\" = {fail_step!r} ]; then echo forced failure; exit 7; fi\n"
        "if [ \"$step\" = freeze ]; then printf model > frozen.pt2; fi\n"
        "if [ \"$step\" = test ]; then printf prediction > pred_train.e.out; fi\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def test_prepare_workspace_counts_frames_and_copies_inputs(tmp_path: Path):
    module = _load_script()
    data = _dataset(tmp_path / "source-data", {"Ag36I36": 2, "Ag18I18": 3})
    model = tmp_path / "base.pt"
    model.write_bytes(b"checkpoint")
    dp = _fake_dp(tmp_path / "dp")
    output = tmp_path / "fine-tune"

    summary = module.prepare_workspace(data, output, model, dp)

    assert summary["systems"] == ["Ag18I18", "Ag36I36"]
    assert summary["total_frames"] == 5
    assert summary["numb_steps"] == 150
    assert summary["step_rule"] == "total_frames_x_30"
    assert (output / "dpa4_pretrained.pt").read_bytes() == b"checkpoint"
    assert not (output / "dpa4_pretrained.pt").is_symlink()
    assert (output / "dpdata" / "Ag36I36" / "set.000" / "coord.npy").is_file()
    assert (output / "run_local.sh").stat().st_mode & 0o111

    expected = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    generated = json.loads((output / "input.json").read_text(encoding="utf-8"))
    expected = copy.deepcopy(expected)
    expected["training"]["numb_steps"] = 150
    assert generated == expected


def test_prepare_override_and_rejects_unsafe_models_or_output(tmp_path: Path):
    module = _load_script()
    data = _dataset(tmp_path / "data", {"Ag": 2})
    model = tmp_path / "base.pt"
    model.write_bytes(b"checkpoint")
    frozen = tmp_path / "base.pt2"
    frozen.write_bytes(b"frozen")
    dp = _fake_dp(tmp_path / "dp")

    summary = module.prepare_workspace(
        data, tmp_path / "override", model, dp, numb_steps=17
    )
    assert summary["numb_steps"] == 17
    assert summary["step_rule"] == "explicit"

    with pytest.raises(module.FineTuningError, match=r"requires a DPA4 \.pt"):
        module.prepare_workspace(data, tmp_path / "bad-model", frozen, dp)
    with pytest.raises(module.FineTuningError, match="already exists"):
        module.prepare_workspace(data, tmp_path / "override", model, dp)
    with pytest.raises(module.FineTuningError, match="inside the source data"):
        module.prepare_workspace(data, data / "nested-output", model, dp)


def test_execute_runs_in_order_and_preserves_failure(tmp_path: Path):
    module = _load_script()
    data = _dataset(tmp_path / "data", {"Ag": 1})
    model = tmp_path / "base.pt"
    model.write_bytes(b"checkpoint")

    successful = tmp_path / "successful"
    module.prepare_workspace(
        data, successful, model, _fake_dp(tmp_path / "dp-success")
    )
    result = module.execute_workspace(successful)
    assert result["status"] == "completed"
    assert (successful / "command-order").read_text().splitlines() == [
        "train",
        "freeze",
        "test",
    ]
    assert (successful / "frozen.pt2").read_bytes() == b"model"
    assert (successful / "pred_train.e.out").is_file()

    failed = tmp_path / "failed"
    module.prepare_workspace(
        data,
        failed,
        model,
        _fake_dp(tmp_path / "dp-fail", fail_step="freeze"),
    )
    with pytest.raises(module.FineTuningError, match="freeze failed"):
        module.execute_workspace(failed)
    failed_summary = json.loads((failed / "run-summary.json").read_text())
    assert failed_summary["status"] == "failed"
    assert failed_summary["failed_step"] == "freeze"
    assert (failed / "command-order").read_text().splitlines() == ["train", "freeze"]
    assert (failed / "freeze_log").is_file()
