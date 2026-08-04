from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from kimodo.resources.cli import run
from kimodo.resources.config import (
    ResourceConfigError,
    ResourceFile,
    ResourceSpec,
    load_catalog,
    load_paths,
)
from kimodo.resources.manager import (
    ResourceManager,
    ResourceVerificationError,
    _unexpected_functional_files,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_catalog(path: Path, *, sha256: str, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
schema_version: 1
groups:
  train-minimal: [tiny]
  optional: [extra]
resources:
  tiny:
    repo_id: example/tiny
    repo_type: model
    revision: 0123456789abcdef0123456789abcdef01234567
    expected_bytes: {size}
    purpose: test resource
    post_fetch: convert later
    files:
      nested/data.bin:
        sha256: {sha256}
        size: {size}
  extra:
    repo_id: example/extra
    repo_type: dataset
    revision: fedcba9876543210fedcba9876543210fedcba98
    opt_in: true
    files:
      extra.bin:
        sha256: {_sha(b'extra')}
        size: 5
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_paths(path: Path, *, destination: str | None, existing: str | None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    destination_yaml = "null" if destination is None else json.dumps(destination)
    existing_yaml = "null" if existing is None else json.dumps(existing)
    path.write_text(
        f"""
schema_version: 1
resources:
  tiny:
    destination: {destination_yaml}
    existing_path: {existing_yaml}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def test_public_catalog_has_pinned_minimal_and_opt_in_resources():
    root = Path(__file__).resolve().parents[2]
    catalog = load_catalog(root / "resources" / "catalog.public.yaml")
    minimal = [item.name for item in catalog.select(["train-minimal"])]
    assert minimal == [
        "bones_seed",
        "kimodo_benchmark",
        "llm2vec_foundation",
        "llm2vec_mntp_adapter",
        "llm2vec_supervised_adapter",
    ]
    assert "qwen_paraphrase_tool" not in minimal
    assert "official_kimodo_seed" not in minimal
    assert catalog.resources["bones_seed"].revision == (
        "2f59b2077b9da34dd4e43618e705c7cb962c9a66"
    )
    assert catalog.resources["kimodo_benchmark"].revision == (
        "2727f526a0c543001d5332b6eabf72d7e3acf14a"
    )
    assert catalog.resources["qwen_paraphrase_tool"].opt_in is True
    assert catalog.resources["official_kimodo_seed"].opt_in is True


def test_llm2vec_snapshot_rejects_unpinned_functional_config(tmp_path):
    spec = ResourceSpec(
        name="llm2vec_supervised_adapter",
        repo_id="example/adapter",
        repo_type="model",
        revision="0" * 40,
        files=(ResourceFile("adapter.bin", "0" * 64, 1),),
    )
    (tmp_path / "llm2vec_config.json").write_text(
        '{"pooling_mode":"last"}\n', encoding="utf-8"
    )
    assert _unexpected_functional_files(spec, tmp_path) == ["llm2vec_config.json"]


def test_catalog_rejects_unpinned_revision_and_unsafe_file(tmp_path):
    catalog_path = _write_catalog(tmp_path / "catalog.yaml", sha256=_sha(b"x"), size=1)
    text = catalog_path.read_text(encoding="utf-8")
    catalog_path.write_text(text.replace("0123456789abcdef0123456789abcdef01234567", "main"))
    with pytest.raises(ResourceConfigError, match="pinned 40-character commit"):
        load_catalog(catalog_path)

    catalog_path = _write_catalog(catalog_path, sha256=_sha(b"x"), size=1)
    text = catalog_path.read_text(encoding="utf-8")
    catalog_path.write_text(text.replace("nested/data.bin", "../data.bin"))
    with pytest.raises(ResourceConfigError, match="safe relative POSIX path"):
        load_catalog(catalog_path)


def test_catalog_rejects_duplicate_keys_and_wrong_expected_bytes(tmp_path):
    catalog_path = _write_catalog(tmp_path / "catalog.yaml", sha256=_sha(b"x"), size=1)
    text = catalog_path.read_text(encoding="utf-8")
    catalog_path.write_text(text.replace("repo_type: model", "repo_type: model\n    repo_type: model"))
    with pytest.raises(ResourceConfigError, match="duplicate YAML key"):
        load_catalog(catalog_path)

    catalog_path = _write_catalog(catalog_path, sha256=_sha(b"x"), size=1)
    catalog_path.write_text(
        catalog_path.read_text(encoding="utf-8").replace("expected_bytes: 1", "expected_bytes: 2")
    )
    with pytest.raises(ResourceConfigError, match="does not equal file sizes"):
        load_catalog(catalog_path)


def test_paths_are_strict_and_resolve_relative_to_yaml(tmp_path):
    catalog = load_catalog(_write_catalog(tmp_path / "catalog.yaml", sha256=_sha(b"x"), size=1))
    paths_file = _write_paths(
        tmp_path / "site" / "paths.yaml", destination="../managed/tiny", existing=None
    )
    paths = load_paths(paths_file, catalog)
    assert paths.binding("tiny").target == (tmp_path / "managed" / "tiny").resolve()

    paths_file.write_text(
        paths_file.read_text(encoding="utf-8").replace(
            "existing_path: null", "existing_path: null\n    token: secret"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ResourceConfigError, match="unknown paths.resources.tiny keys"):
        load_paths(paths_file, catalog)


def test_existing_path_is_verified_and_never_downloaded_or_modified(tmp_path):
    payload = b"verified"
    catalog = load_catalog(
        _write_catalog(tmp_path / "catalog.yaml", sha256=_sha(payload), size=len(payload))
    )
    existing = tmp_path / "shared"
    (existing / "nested").mkdir(parents=True)
    (existing / "nested" / "data.bin").write_bytes(payload)
    paths = load_paths(
        _write_paths(tmp_path / "paths.yaml", destination=None, existing=str(existing)), catalog
    )
    before = sorted(path.relative_to(existing) for path in existing.rglob("*"))

    def forbidden_download(**_kwargs):
        raise AssertionError("existing_path must not call the downloader")

    manager = ResourceManager(catalog, paths, downloader=forbidden_download)
    result = manager.fetch(["train-minimal"])
    assert result["status"] == "fetched_and_verified"
    assert result["resources"][0]["mode"] == "existing"
    assert sorted(path.relative_to(existing) for path in existing.rglob("*")) == before


def test_corrupt_existing_path_fails_closed_without_download(tmp_path):
    catalog = load_catalog(
        _write_catalog(tmp_path / "catalog.yaml", sha256=_sha(b"good"), size=4)
    )
    existing = tmp_path / "shared"
    (existing / "nested").mkdir(parents=True)
    (existing / "nested" / "data.bin").write_bytes(b"evil")
    paths = load_paths(
        _write_paths(tmp_path / "paths.yaml", destination=None, existing=str(existing)), catalog
    )
    manager = ResourceManager(
        catalog,
        paths,
        downloader=lambda **_kwargs: pytest.fail("must not download into existing_path"),
    )
    with pytest.raises(ResourceVerificationError, match="never modifies existing_path"):
        manager.fetch(["train-minimal"])
    assert (existing / "nested" / "data.bin").read_bytes() == b"evil"


def test_managed_fetch_pins_revision_verifies_and_resumes(tmp_path):
    payload = b"downloaded"
    catalog = load_catalog(
        _write_catalog(tmp_path / "catalog.yaml", sha256=_sha(payload), size=len(payload))
    )
    destination = tmp_path / "managed"
    paths = load_paths(
        _write_paths(tmp_path / "paths.yaml", destination=str(destination), existing=None), catalog
    )
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return str(target)

    manager = ResourceManager(catalog, paths, downloader=download)
    first = manager.fetch(["train-minimal"])
    assert first["resources"][0]["downloaded"] == ["nested/data.bin"]
    assert calls == [
        {
            "repo_id": "example/tiny",
            "filename": "nested/data.bin",
            "repo_type": "model",
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "local_dir": str(destination),
            "force_download": False,
            "local_files_only": False,
        }
    ]
    assert "token" not in calls[0]
    receipt = destination / ".cache" / "kimodo" / "resource-receipt.json"
    assert receipt.is_file()
    assert "token" not in receipt.read_text(encoding="utf-8").lower()

    calls.clear()
    second = manager.fetch(["train-minimal"])
    assert second["resources"][0]["downloaded"] == []
    assert second["resources"][0]["reused"] == ["nested/data.bin"]
    assert calls == []


def test_plan_is_read_only_and_cli_default_group_excludes_optional(tmp_path, capsys):
    payload = b"x"
    catalog_file = _write_catalog(tmp_path / "catalog.yaml", sha256=_sha(payload), size=1)
    paths_file = _write_paths(
        tmp_path / "paths.yaml", destination=str(tmp_path / "managed"), existing=None
    )
    catalog = load_catalog(catalog_file)
    paths = load_paths(paths_file, catalog)
    manager = ResourceManager(
        catalog, paths, downloader=lambda **_kwargs: pytest.fail("plan must not download")
    )
    plan = manager.plan(["train-minimal"])
    assert plan["resources"][0]["network_required"] is True
    assert not (tmp_path / "managed").exists()

    assert run(["--catalog", str(catalog_file), "--paths", str(paths_file), "plan"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["groups"] == ["train-minimal"]
    assert [item["name"] for item in output["resources"]] == ["tiny"]
