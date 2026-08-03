from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch

from kimodo.model.llm2vec.llm2vec import LLM2Vec
from kimodo.model.llm2vec.llm2vec_wrapper import LLM2VecEncoder
from kimodo.training import text_cache_cli


class _DummyTokenizer:
    eos_token = "<eos>"
    pad_token = None
    padding_side = "right"


class LlamaConfig:
    pass


class _DummyModel(torch.nn.Module):
    def __init__(self, name: str = "unset") -> None:
        super().__init__()
        self.config = SimpleNamespace(_name_or_path=name)


def test_explicit_three_layer_loader_keeps_revisions_on_their_own_repos():
    tokenizer = _DummyTokenizer()
    config = LlamaConfig()
    config._name_or_path = "mntp"
    foundation = _DummyModel("foundation-local")
    merged_mntp = _DummyModel("merged-mntp")
    supervised = _DummyModel("supervised")
    mntp_adapter = MagicMock()
    mntp_adapter.merge_and_unload.return_value = merged_mntp
    model_class = MagicMock()
    model_class.from_pretrained.return_value = foundation

    with (
        patch("kimodo.model.llm2vec.llm2vec.AutoTokenizer.from_pretrained", return_value=tokenizer) as tokenizer_load,
        patch("kimodo.model.llm2vec.llm2vec.AutoConfig.from_pretrained", return_value=config) as config_load,
        patch.object(LLM2Vec, "_get_model_class", return_value=model_class),
        patch("kimodo.model.llm2vec.llm2vec.PeftConfig.from_pretrained") as peft_config_load,
        patch(
            "kimodo.model.llm2vec.llm2vec.PeftModel.from_pretrained",
            side_effect=[mntp_adapter, supervised],
        ) as peft_load,
    ):
        peft_config_load.return_value = SimpleNamespace(
            base_model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct"
        )
        encoder = LLM2Vec.from_pretrained(
            base_model_name_or_path="mntp-local",
            peft_model_name_or_path="supervised-local",
            foundation_model_name_or_path="foundation-local",
            base_revision="mntp-sha",
            peft_revision="supervised-sha",
            foundation_revision="foundation-sha",
            torch_dtype=torch.float32,
            cache_dir="/cache",
        )

    tokenizer_load.assert_called_once_with("mntp-local", revision="mntp-sha", cache_dir="/cache")
    config_load.assert_called_once_with("mntp-local", revision="mntp-sha", cache_dir="/cache")
    peft_config_load.assert_called_once_with("mntp-local", revision="mntp-sha", cache_dir="/cache")
    model_class.from_pretrained.assert_called_once_with(
        "foundation-local",
        revision="foundation-sha",
        config=config,
        torch_dtype=torch.float32,
        cache_dir="/cache",
    )
    assert peft_load.call_args_list == [
        call(foundation, "mntp-local", revision="mntp-sha", cache_dir="/cache"),
        call(merged_mntp, "supervised-local", revision="supervised-sha", cache_dir="/cache"),
    ]
    mntp_adapter.merge_and_unload.assert_called_once_with()
    assert foundation.config._name_or_path == "meta-llama/Meta-Llama-3-8B-Instruct"
    assert encoder.model is supervised


def test_legacy_two_layer_loader_remains_available_for_released_inference_defaults():
    tokenizer = _DummyTokenizer()
    config = LlamaConfig()
    model = _DummyModel("meta-llama/Meta-Llama-3-8B-Instruct")
    model_class = MagicMock()
    model_class.from_pretrained.return_value = model

    with (
        patch("kimodo.model.llm2vec.llm2vec.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("kimodo.model.llm2vec.llm2vec.AutoConfig.from_pretrained", return_value=config),
        patch.object(LLM2Vec, "_get_model_class", return_value=model_class),
        patch("kimodo.model.llm2vec.llm2vec.PeftConfig.from_pretrained") as peft_config_load,
    ):
        encoder = LLM2Vec.from_pretrained(
            base_model_name_or_path="mntp-repo",
            base_revision=None,
            torch_dtype=torch.bfloat16,
        )

    model_class.from_pretrained.assert_called_once_with(
        "mntp-repo",
        revision=None,
        torch_dtype=torch.bfloat16,
    )
    peft_config_load.assert_not_called()
    assert encoder.model is model


def test_encoder_wrapper_resolves_and_forwards_all_three_local_snapshots(monkeypatch):
    monkeypatch.setenv("TEXT_ENCODERS_DIR", "/snapshots")
    monkeypatch.delenv("HUGGINGFACE_CACHE_DIR", raising=False)
    model = _DummyModel("meta-llama/Meta-Llama-3-8B-Instruct")
    with patch(
        "kimodo.model.llm2vec.llm2vec_wrapper.LLM2Vec.from_pretrained",
        return_value=model,
    ) as load:
        encoder = LLM2VecEncoder(
            foundation_model_name_or_path="foundation",
            base_model_name_or_path="mntp",
            peft_model_name_or_path="supervised",
            foundation_revision="foundation-sha",
            base_revision="mntp-sha",
            peft_revision="supervised-sha",
            dtype="float32",
            llm_dim=4096,
            device="cpu",
        )

    load.assert_called_once_with(
        foundation_model_name_or_path="/snapshots/foundation",
        base_model_name_or_path="/snapshots/mntp",
        peft_model_name_or_path="/snapshots/supervised",
        foundation_revision="foundation-sha",
        base_revision="mntp-sha",
        peft_revision="supervised-sha",
        torch_dtype=torch.float32,
        cache_dir=None,
    )
    assert encoder.model is model
    assert not model.training


def _local_cache_args(tmp_path) -> argparse.Namespace:
    return argparse.Namespace(
        provider="local",
        api_url="http://127.0.0.1:9550/",
        device="cpu",
        foundation_model="/models/foundation",
        foundation_revision="foundation-sha",
        mntp_model="/models/mntp",
        mntp_revision="mntp-sha",
        supervised_model="/models/supervised",
        supervised_revision="supervised-sha",
        manifest=str(tmp_path / "source.jsonl"),
        output_manifest=str(tmp_path / "cached.jsonl"),
        cache_dir=str(tmp_path / "cache"),
    )


def test_text_cache_identity_and_sidecar_record_all_three_artifacts(tmp_path):
    args = _local_cache_args(tmp_path)
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"id": "one", "text": "A person walks."}) + "\n", encoding="utf-8")
    fake_encoder = MagicMock(return_value=(torch.ones(1, 1, 4096), [1]))

    with patch.object(
        text_cache_cli,
        "_build_encoder",
        return_value=(
            fake_encoder,
            "llm2vec:foundation=/models/foundation@foundation-sha;"
            "mntp=/models/mntp@mntp-sha;"
            "supervised=/models/supervised@supervised-sha;"
            "pooling=mean;dtype=float32;internal_batch_size=1",
        ),
    ):
        text_cache_cli.run(args)

    sidecar = json.loads((tmp_path / "cached.jsonl.metadata.json").read_text(encoding="utf-8"))
    assert sidecar["schema_version"] == 2
    assert sidecar["encoder_artifacts"] == {
        "foundation": {
            "model_name_or_path": "/models/foundation",
            "revision": "foundation-sha",
        },
        "mntp_adapter": {
            "model_name_or_path": "/models/mntp",
            "revision": "mntp-sha",
        },
        "supervised_adapter": {
            "model_name_or_path": "/models/supervised",
            "revision": "supervised-sha",
        },
    }
    cached = json.loads((tmp_path / "cached.jsonl").read_text(encoding="utf-8"))
    expected_key = text_cache_cli._cache_key("A person walks.", sidecar["encoder"])
    assert cached["text_cache_key"] == expected_key
    assert (tmp_path / "cache" / f"{expected_key}.npy").is_file()


def test_text_cache_cli_accepts_new_names_and_legacy_adapter_aliases():
    parser = text_cache_cli.build_parser()
    args = parser.parse_args(
        [
            "--manifest",
            "source.jsonl",
            "--output-manifest",
            "cached.jsonl",
            "--cache-dir",
            "cache",
            "--foundation-revision",
            "foundation-sha",
            "--base-revision",
            "mntp-sha",
            "--peft-revision",
            "supervised-sha",
        ]
    )
    assert args.foundation_model == "meta-llama/Meta-Llama-3-8B-Instruct"
    assert args.mntp_revision == "mntp-sha"
    assert args.supervised_revision == "supervised-sha"
