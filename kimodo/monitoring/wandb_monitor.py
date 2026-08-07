"""Best-effort Weights & Biases monitoring without changing training semantics."""

from __future__ import annotations

import hashlib
import importlib
import os
import re
import sys
from pathlib import Path
from typing import Any


_TRUE = {"1", "true", "yes", "on"}
_SECRET_TOKENS = ("api_key", "apikey", "password", "secret", "token")


def _truthy(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUE


def _monitoring_enabled() -> bool:
    explicit = os.environ.get("KIMODO_WANDB_ENABLED")
    if explicit is not None:
        return explicit.strip().lower() in _TRUE
    return bool(os.environ.get("WANDB_API_KEY")) or os.environ.get("WANDB_MODE") == "offline"


def _run_identity(path: str | Path) -> str:
    resolved = str(Path(path).expanduser().resolve())
    name = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(resolved).name).strip("-") or "kimodo"
    return f"{name[:80]}-{hashlib.sha256(resolved.encode('utf-8')).hexdigest()[:8]}"


def _scope_env(scope: str, suffix: str) -> str | None:
    return os.environ.get(f"KIMODO_WANDB_{scope.upper()}_{suffix}") or os.environ.get(
        f"KIMODO_WANDB_{suffix}"
    )


def _safe_value(value: Any, key: str = "") -> Any:
    """Convert configuration to W&B-safe values and redact accidental secrets."""
    if any(token in key.lower() for token in _SECRET_TOKENS):
        return "<redacted>"
    if isinstance(value, dict):
        return {str(child): _safe_value(item, str(child)) for child, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class WandbMonitor:
    """Small failure-isolated wrapper around a lazily imported W&B run."""

    def __init__(self, run=None, *, required: bool = False) -> None:
        self.run = run
        self.required = required
        self._warned = False

    @property
    def enabled(self) -> bool:
        return self.run is not None

    @classmethod
    def from_env(
        cls,
        scope: str,
        *,
        output_dir: str | Path,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        active: bool = True,
        identity_root: str | Path | None = None,
    ) -> "WandbMonitor":
        required = _truthy("KIMODO_WANDB_REQUIRED")
        if not active or not _monitoring_enabled():
            return cls(required=required)
        try:
            wandb = importlib.import_module("wandb")
            output = Path(output_dir).expanduser().resolve()
            output.mkdir(parents=True, exist_ok=True)
            identity = _run_identity(identity_root or output.parent)
            init_args: dict[str, Any] = {
                "project": os.environ.get("WANDB_PROJECT", "kimodo-reproduction"),
                "job_type": scope,
                "dir": str(output),
                "config": _safe_value(config or {}),
                "reinit": True,
            }
            optional = {
                "entity": os.environ.get("WANDB_ENTITY"),
                "group": os.environ.get("WANDB_RUN_GROUP")
                or os.environ.get("KIMODO_WANDB_GROUP")
                or identity,
                "name": _scope_env(scope, "RUN_NAME") or f"{identity}-{scope}",
                "id": _scope_env(scope, "RUN_ID") or f"{identity}-{scope}",
            }
            init_args.update({key: value for key, value in optional.items() if value})
            init_args["resume"] = os.environ.get("WANDB_RESUME", "allow")
            tags = _scope_env(scope, "TAGS")
            if tags:
                init_args["tags"] = [tag.strip() for tag in tags.split(",") if tag.strip()]
            run = wandb.init(**init_args)
            if run is None:
                raise RuntimeError("wandb.init returned no run")
            run.define_metric("global_step")
            run.define_metric("*", step_metric="global_step")
            if metadata:
                run.summary.update(_safe_value(metadata))
            return cls(run, required=required)
        except Exception as error:
            if required:
                raise RuntimeError(f"required W&B initialization failed for {scope}: {error}") from error
            print(
                f"Kimodo W&B monitoring disabled after initialization failure: {error}",
                file=sys.stderr,
                flush=True,
            )
            return cls(required=False)

    def _failure(self, operation: str, error: Exception) -> None:
        if self.required:
            raise RuntimeError(f"required W&B {operation} failed: {error}") from error
        if not self._warned:
            print(
                f"Kimodo W&B monitoring stopped after {operation} failure: {error}",
                file=sys.stderr,
                flush=True,
            )
            self._warned = True
        self.run = None

    def log(self, record: dict[str, Any], *, step: int | None = None) -> None:
        if self.run is None:
            return
        payload = dict(record)
        if step is not None:
            payload["global_step"] = int(step)
        try:
            self.run.log(_safe_value(payload))
        except Exception as error:
            self._failure("log", error)

    def summary(self, values: dict[str, Any]) -> None:
        if self.run is None:
            return
        try:
            self.run.summary.update(_safe_value(values))
        except Exception as error:
            self._failure("summary update", error)

    def finish(self, *, exit_code: int = 0) -> None:
        if self.run is None:
            return
        run, self.run = self.run, None
        try:
            run.finish(exit_code=exit_code)
        except Exception as error:
            if self.required:
                raise RuntimeError(f"required W&B finish failed: {error}") from error
            print(f"Kimodo W&B finish failed: {error}", file=sys.stderr, flush=True)
