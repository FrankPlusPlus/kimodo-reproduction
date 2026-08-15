#!/usr/bin/env python3
"""Poll a live train.jsonl and email if loss/grad looks like the kf=8 collapse.

Intended to run on the company 开发机 (transfer-test), which can read the PVC
run directory. SMTP settings come from the environment / PVC .env — do not put
the QQ authorization code in git.

  export KIMODO_ALERT_SMTP_PASSWORD='qq-smtp-auth-code'
  bash scripts/watch_train_loss_alert.sh
"""

from __future__ import annotations

import argparse
import json
import math
import os
import smtplib
import ssl
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from email.message import EmailMessage
from pathlib import Path
from statistics import median
from typing import Any


CST = timezone(timedelta(hours=8))

DEFAULT_RUN_DIR = (
    "/home/share/yezitao-kimodo-reproduction/runs/v2-1m-hostnet-kf-smooth-lr1e5"
)
DEFAULT_TO = "171024830@qq.com"
ENV_FILES = (
    os.environ.get("KIMODO_ENV_FILE"),
    "/home/share/yzt/kimodo-reproduction/.env",
    "/home/share/yezitao-kimodo-reproduction/secrets/kimodo.env",
)


def load_env_files() -> None:
    """Fill empty process env from PVC .env so SMTP/LLM keys can appear later."""
    for raw in ENV_FILES:
        if not raw:
            continue
        path = Path(raw)
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[7:].strip()
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            if key.isidentifier() and not os.environ.get(key):
                os.environ[key] = value


@dataclass(frozen=True)
class Alert:
    reason: str
    summary: str


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def evaluate_record(record: dict[str, Any], recent_losses: list[float]) -> list[Alert]:
    """Return alerts for one metrics line. recent_losses are prior finite totals."""
    alerts: list[Alert] = []
    step = record.get("global_step")
    loss = record.get("loss/total")
    loss_value = _finite(loss)
    gnorm = _finite(record.get("optimizer/gradient_norm_before_clip"))
    clip_frac = _finite(record.get("optimizer/gradient_clip_fraction"))
    skip_frac = _finite(record.get("optimizer/extreme_gradient_skip_fraction"))

    if loss is not None and loss_value is None:
        alerts.append(Alert("nonfinite_loss", f"step={step} loss/total={loss!r} is not finite"))
    elif loss_value is not None and loss_value >= 1.0:
        alerts.append(
            Alert(
                "loss_absolute",
                f"step={step} loss/total={loss_value:.4f} >= 1.0 "
                "(healthy kf-smooth window is ~0.24)",
            )
        )
    elif loss_value is not None and len(recent_losses) >= 10:
        baseline = float(median(recent_losses[-30:]))
        if baseline > 0 and loss_value >= max(0.8, 3.0 * baseline):
            alerts.append(
                Alert(
                    "loss_spike",
                    f"step={step} loss/total={loss_value:.4f} is "
                    f"{loss_value / baseline:.1f}x the recent median {baseline:.4f}",
                )
            )

    if gnorm is not None and gnorm >= 8.0:
        alerts.append(
            Alert(
                "grad_norm",
                f"step={step} gradient_norm_before_clip={gnorm:.3f} >= 8 "
                "(skip fuse is 5; pre-collapse climb was tens to hundreds)",
            )
        )
    if clip_frac is not None and clip_frac >= 0.5:
        alerts.append(
            Alert(
                "clip_saturated",
                f"step={step} gradient_clip_fraction={clip_frac:.3f} >= 0.5",
            )
        )
    if skip_frac is not None and skip_frac >= 0.05:
        alerts.append(
            Alert(
                "extreme_skip",
                f"step={step} extreme_gradient_skip_fraction={skip_frac:.3f} >= 0.05",
            )
        )
    return alerts


def format_record_lines(record: dict[str, Any]) -> list[str]:
    keys = (
        "global_step",
        "epoch",
        "loss/total",
        "optimizer/learning_rate",
        "optimizer/gradient_norm_before_clip",
        "optimizer/gradient_clip_fraction",
        "optimizer/extreme_gradient_skip_fraction",
        "system/optimizer_steps_per_second",
    )
    lines = []
    for key in keys:
        if key in record:
            lines.append(f"{key}={record[key]}")
    return lines


def build_email(
    *,
    to_addr: str,
    from_addr: str,
    run_dir: Path,
    alerts: list[Alert],
    record: dict[str, Any] | None,
    extra: str = "",
) -> EmailMessage:
    now = datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S %Z")
    reasons = ", ".join(alert.reason for alert in alerts)
    step = None if record is None else record.get("global_step")
    subject = f"[kimodo] train alert step={step} ({reasons})"
    body = [
        f"time: {now}",
        f"run: {run_dir}",
        "",
        "alerts:",
        *[f"- {alert.reason}: {alert.summary}" for alert in alerts],
    ]
    if extra:
        body.extend(["", extra])
    if record is not None:
        body.extend(["", "latest metrics:", *format_record_lines(record)])
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = from_addr
    message["To"] = to_addr
    message.set_content("\n".join(body) + "\n")
    return message


def send_email(message: EmailMessage) -> None:
    load_env_files()
    host = os.environ.get("KIMODO_ALERT_SMTP_HOST", "smtp.qq.com")
    port = int(os.environ.get("KIMODO_ALERT_SMTP_PORT", "465"))
    user = os.environ.get("KIMODO_ALERT_SMTP_USER", message["From"])
    password = os.environ.get("KIMODO_ALERT_SMTP_PASSWORD", "")
    if not password:
        raise RuntimeError(
            "KIMODO_ALERT_SMTP_PASSWORD is empty; use the QQ mailbox SMTP 授权码, "
            "not the login password"
        )
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(host, port, timeout=30, context=context) as smtp:
        smtp.login(user, password)
        smtp.send_message(message)


def load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"offset": 0, "recent_losses": [], "last_alert_unix": {}, "last_step": None}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def parse_new_records(handle, offset: int) -> tuple[int, list[dict[str, Any]]]:
    handle.seek(offset)
    records: list[dict[str, Any]] = []
    while True:
        line = handle.readline()
        if line == "":
            break
        if not line.endswith("\n"):
            # Incomplete last line; wait for the next poll.
            handle.seek(offset)
            break
        offset = handle.tell()
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return offset, records


def maybe_alert(
    *,
    alerts: list[Alert],
    state: dict[str, Any],
    cooldown_s: float,
    now: float,
) -> list[Alert]:
    last = state.setdefault("last_alert_unix", {})
    due: list[Alert] = []
    for alert in alerts:
        previous = float(last.get(alert.reason, 0.0))
        if now - previous >= cooldown_s:
            due.append(alert)
            last[alert.reason] = now
    return due


def watch_once(
    *,
    jsonl: Path,
    state: dict[str, Any],
    stall_s: float,
    now: float,
) -> tuple[list[tuple[list[Alert], dict[str, Any] | None]], dict[str, Any]]:
    events: list[tuple[list[Alert], dict[str, Any] | None]] = []
    if not jsonl.is_file():
        return events, state
    with jsonl.open("r", encoding="utf-8") as handle:
        offset, records = parse_new_records(handle, int(state.get("offset", 0)))
    state["offset"] = offset
    recent = [float(value) for value in state.get("recent_losses", [])]
    last_record = None
    for record in records:
        alerts = evaluate_record(record, recent)
        if alerts:
            events.append((alerts, record))
        loss_value = _finite(record.get("loss/total"))
        if loss_value is not None:
            recent.append(loss_value)
            recent = recent[-60:]
        if record.get("global_step") is not None:
            state["last_step"] = record.get("global_step")
        last_record = record
        state["last_record_unix"] = now
    state["recent_losses"] = recent
    last_seen = float(state.get("last_record_unix") or 0.0)
    if last_seen and now - last_seen >= stall_s:
        events.append(
            (
                [
                    Alert(
                        "stalled",
                        f"train.jsonl has not grown for {int(now - last_seen)}s "
                        f"(last_step={state.get('last_step')})",
                    )
                ],
                last_record,
            )
        )
    return events, state


def log(message: str) -> None:
    stamp = datetime.now(CST).strftime("%H:%M:%S")
    print(f"[loss-watch {stamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(os.environ.get("KIMODO_RUN_DIR", DEFAULT_RUN_DIR)),
    )
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--cooldown-seconds", type=float, default=1200.0)
    parser.add_argument("--stall-seconds", type=float, default=720.0)
    parser.add_argument("--to", default=os.environ.get("KIMODO_ALERT_EMAIL", DEFAULT_TO))
    parser.add_argument(
        "--from-addr",
        default=os.environ.get("KIMODO_ALERT_SMTP_USER", DEFAULT_TO),
    )
    parser.add_argument("--send-test", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--log-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    jsonl = run_dir / "train.jsonl"
    watch_dir = Path(str(run_dir) + ".loss-watch")
    state_path = watch_dir / "state.json"
    watch_dir.mkdir(parents=True, exist_ok=True)

    if args.send_test:
        message = build_email(
            to_addr=args.to,
            from_addr=args.from_addr,
            run_dir=run_dir,
            alerts=[Alert("test", "SMTP test from watch_train_loss_alert.py")],
            record=None,
            extra="If you received this, loss alerts can reach the mailbox.",
        )
        send_email(message)
        log(f"sent test email to {args.to}")
        return 0

    if not args.log_only and not os.environ.get("KIMODO_ALERT_SMTP_PASSWORD"):
        log("KIMODO_ALERT_SMTP_PASSWORD is missing; refusing to watch without mail")
        return 2

    log(f"watching {jsonl} -> {args.to} poll={args.poll_seconds}s")
    state = load_state(state_path)
    while True:
        now = time.time()
        events, state = watch_once(
            jsonl=jsonl,
            state=state,
            stall_s=args.stall_seconds,
            now=now,
        )
        for alerts, record in events:
            due = maybe_alert(
                alerts=alerts,
                state=state,
                cooldown_s=args.cooldown_seconds,
                now=now,
            )
            if not due:
                continue
            log("; ".join(alert.summary for alert in due))
            if not args.log_only:
                message = build_email(
                    to_addr=args.to,
                    from_addr=args.from_addr,
                    run_dir=run_dir,
                    alerts=due,
                    record=record,
                )
                try:
                    send_email(message)
                    log(f"emailed {args.to}")
                except Exception as error:  # noqa: BLE001 - keep watching after SMTP blips
                    log(f"SMTP send failed: {error}")
                    for alert in due:
                        state.setdefault("last_alert_unix", {}).pop(alert.reason, None)
        save_state(state_path, state)
        if args.once:
            return 0
        time.sleep(max(1.0, args.poll_seconds))


if __name__ == "__main__":
    sys.exit(main())
