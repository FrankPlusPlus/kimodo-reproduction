from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from kimodo.model.load_model import TEXT_ENCODER_PRESETS, _build_local_text_encoder_conf


def test_local_text_encoder_paths_are_explicit_and_do_not_mutate_preset(tmp_path, monkeypatch):
    foundation = tmp_path / "foundation"
    mntp = tmp_path / "mntp"
    supervised = tmp_path / "supervised"
    for path in (foundation, mntp, supervised):
        path.mkdir()

    monkeypatch.setenv("KIMODO_LLM2VEC_FOUNDATION", str(foundation))
    monkeypatch.setenv("KIMODO_LLM2VEC_MNTP", str(mntp))
    monkeypatch.setenv("KIMODO_LLM2VEC_SUPERVISED", str(supervised))
    config = _build_local_text_encoder_conf(text_encoder_fp32=True)

    assert config["foundation_model_name_or_path"] == str(foundation.resolve())
    assert config["base_model_name_or_path"] == str(mntp.resolve())
    assert config["peft_model_name_or_path"] == str(supervised.resolve())
    assert config["dtype"] == "float32"
    assert TEXT_ENCODER_PRESETS["llm2vec"]["kwargs"]["dtype"] == "bfloat16"


def test_local_text_encoder_paths_fail_closed_when_incomplete(tmp_path, monkeypatch):
    foundation = tmp_path / "foundation"
    foundation.mkdir()
    monkeypatch.setenv("KIMODO_LLM2VEC_FOUNDATION", str(foundation))

    with pytest.raises(ValueError, match="all three path variables"):
        _build_local_text_encoder_conf()


def test_benchmark_generation_imports_callable_loaders():
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "benchmark" / "generate_eval.py")
    )

    assert callable(namespace["load_checkpoint_bundle"])
    assert callable(namespace["load_model"])
