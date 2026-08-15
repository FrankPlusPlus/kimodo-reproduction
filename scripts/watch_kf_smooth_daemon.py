#!/usr/bin/env python3
"""Long-running 开发机 daemon: loss email alerts + 10k health + 50k eval reports.

This host has no GPU. It will not run the benchmark itself. It emails:
  - loss/grad collapse alerts from train.jsonl
  - a gradient/health snapshot every 10k from 610k
  - a benchmark-only analysis when stratified-10pct summary_rows.json appears
    at each 50k from 650k (same metrics as the previous 100k eval mails)
  - one Official SEED-v1.1 vs 695k head-to-head once both summaries exist

SMTP 授权码 can be added to PVC .env after start; the loop reloads empty env keys.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from statistics import mean, median
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import watch_train_loss_alert as watch  # noqa: E402


DEFAULT_EVAL_ROOT = (
    "/home/share/yezitao-kimodo-reproduction/eval-results/"
    "v2-1m-hostnet-cap800-from695k-stratified10pct"
)
DEFAULT_PRIOR_EVAL_ROOT = (
    "/home/share/yezitao-kimodo-reproduction/eval-results/"
    "v2-1m-hostnet-kf-smooth-lr1e5-stratified10pct"
)
DEFAULT_OFFICIAL_EVAL_SUMMARY = (
    "/home/share/yezitao-kimodo-reproduction/eval-results/"
    "official-seed-v1-stratified10pct/summary_rows.json"
)
DEFAULT_695K_EVAL_SUMMARY = (
    "/home/share/yezitao-kimodo-reproduction/eval-results/"
    "v2-1m-hostnet-kf-smooth-lr1e5-step695k-stratified10pct/"
    "step-000695000/summary_rows.json"
)
CONTEXT_EVAL_SUMMARIES = (
    (
        "600k original",
        "/home/share/yezitao-kimodo-reproduction/eval-results/"
        "v2-1m-hostnet-stratified10pct/step-000600000/summary_rows.json",
    ),
    (
        "650k kf-smooth",
        "/home/share/yezitao-kimodo-reproduction/eval-results/"
        "v2-1m-hostnet-kf-smooth-lr1e5-stratified10pct/"
        "step-000650000/summary_rows.json",
    ),
)
FORK_BASELINE_STEP = 695_000
MILESTONES = tuple(range(700_000, 1_000_001, 50_000))
HEALTH_10K_START = 700_000
HEALTH_10K_EVERY = 10_000
CONTENT_KEYS = (
    "Full-Body Pos (gen, cm)",
    "End-Effector Pos (gen, cm)",
    "2D Root Pos (gen, cm)",
    "Skate (gen, cm/s)",
    "FID gen-GT",
    "FID gen-text",
    "R@3 (gen)",
    "Contact (gen)",
)
CONSTRAINT_KEYS = (
    "Full-Body Pos (gen, cm)",
    "End-Effector Pos (gen, cm)",
    "2D Root Pos (gen, cm)",
)
PRIOR_KEYS = (
    "FID gen-GT",
    "FID gen-text",
    "R@3 (gen)",
    "Contact (gen)",
)
SKATE_KEY = "Skate (gen, cm/s)"
HIGHER_BETTER = {"R@3 (gen)", "Contact (gen)"}
DELTA_EPS = {
    "Full-Body Pos (gen, cm)": 0.5,
    "End-Effector Pos (gen, cm)": 0.5,
    "2D Root Pos (gen, cm)": 0.5,
    "Skate (gen, cm/s)": 0.15,
    "FID gen-GT": 0.01,
    "FID gen-text": 0.01,
    "R@3 (gen)": 2.0,
    "Contact (gen)": 0.01,
}


def _walk_numbers(obj: Any, prefix: str = "") -> list[tuple[str, float]]:
    found: list[tuple[str, float]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}/{key}" if prefix else str(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                found.append((path, float(value)))
            else:
                found.extend(_walk_numbers(value, path))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            found.extend(_walk_numbers(value, f"{prefix}[{index}]"))
    return found


def extract_eval_highlights(
    summary: dict[str, Any], split: str = "content"
) -> dict[str, float]:
    highlights: dict[str, float] = {}
    tables = summary.get("tables") or {}
    block = tables.get(split) if isinstance(tables, dict) else None
    if not isinstance(block, dict):
        return highlights
    preferred: list[tuple[str, float]] = []
    fallback: list[tuple[str, float]] = []
    for path, value in _walk_numbers(block, split):
        for key in CONTENT_KEYS:
            if key not in path:
                continue
            if "text_following[0]" in path or "constraints[0]" in path:
                preferred.append((key, value))
            else:
                fallback.append((key, value))
    for key, value in preferred:
        highlights.setdefault(key, value)
    for key, value in fallback:
        highlights.setdefault(key, value)
    return highlights


def classify_delta(key: str, current: float, previous: float) -> str:
    eps = DELTA_EPS.get(key, 0.0)
    if key in HIGHER_BETTER:
        change = current - previous
    else:
        change = previous - current
    if change > eps:
        return "进步"
    if change < -eps:
        return "退步"
    return "持平"


def compare_eval_highlights(
    current: dict[str, float],
    previous: dict[str, float],
    *,
    prior_step: int,
    milestone: int,
    title: str | None = None,
) -> str:
    lines = [title or f"Phase 2 对照 {prior_step} -> {milestone}"]

    def _block(title: str, keys: tuple[str, ...], *, prior_watch: bool) -> None:
        lines.append(title)
        for key in keys:
            if key not in current:
                continue
            cur = current[key]
            prev = previous.get(key)
            if prev is None:
                lines.append(f"  {key}: {cur:.4f} (无上一档)")
                continue
            verdict = classify_delta(key, cur, prev)
            if prior_watch:
                label = "退步" if verdict == "退步" else "未退步"
            else:
                label = verdict
            delta = cur - prev
            lines.append(f"  {key}: {prev:.4f} -> {cur:.4f} ({delta:+.4f}) {label}")

    _block("约束（主看，越低越好，对照官方约 4cm）：", CONSTRAINT_KEYS, prior_watch=False)
    _block("足滑（看有没有进步）：", (SKATE_KEY,), prior_watch=False)
    _block("先验（FID/R@3/接触，只看有没有退步）：", PRIOR_KEYS, prior_watch=True)
    return "\n".join(lines)


def format_highlights(title: str, highlights: dict[str, float]) -> str:
    if not highlights:
        return f"{title}: (none)"
    lines = [title]
    for key in CONTENT_KEYS:
        if key in highlights:
            lines.append(f"  {key}: {highlights[key]:.4f}")
    return "\n".join(lines)


def fmt_stat(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def scheduled_kmax(step: int) -> float:
    progress = min(1.0, max(0.0, (step - 500_000) / 499_999))
    return 1.0 + progress * 19.0


def slice_window(records: list[dict[str, Any]], center: int, width: int) -> list[dict[str, Any]]:
    low = center - width
    return [
        row
        for row in records
        if isinstance(row.get("global_step"), int) and low <= int(row["global_step"]) <= center
    ]


def _values(rows: list[dict[str, Any]], key: str) -> list[float]:
    found: list[float] = []
    for row in rows:
        value = watch._finite(row.get(key))
        if value is not None:
            found.append(value)
    return found


def summarize_window(window: list[dict[str, Any]]) -> dict[str, float | None]:
    losses = _values(window, "loss/total")
    gnorms = _values(window, "optimizer/gradient_norm_before_clip")
    clips = _values(window, "optimizer/gradient_clip_fraction")
    skips = _values(window, "optimizer/extreme_gradient_skip_fraction")
    return {
        "loss_mean": mean(losses) if losses else None,
        "loss_median": median(losses) if losses else None,
        "loss_max": max(losses) if losses else None,
        "gnorm_mean": mean(gnorms) if gnorms else None,
        "gnorm_median": median(gnorms) if gnorms else None,
        "gnorm_max": max(gnorms) if gnorms else None,
        "clip_max": max(clips) if clips else None,
        "skip_max": max(skips) if skips else None,
    }


def health_status(stats: dict[str, float | None]) -> str:
    loss_med = stats.get("loss_median")
    g_med = stats.get("gnorm_median")
    g_max = stats.get("gnorm_max")
    clip_max = stats.get("clip_max")
    skip_max = stats.get("skip_max")
    if (
        (loss_med is not None and loss_med >= 1.0)
        or (g_max is not None and g_max >= 8.0)
        or (clip_max is not None and clip_max >= 0.5)
        or (skip_max is not None and skip_max >= 0.05)
    ):
        return "异常"
    if (
        (loss_med is not None and loss_med >= 0.4)
        or (g_med is not None and g_med >= 1.0)
        or (g_max is not None and g_max >= 5.0)
        or (clip_max is not None and clip_max >= 0.1)
        or (skip_max is not None and skip_max >= 0.01)
    ):
        return "观察"
    if loss_med is not None and 0.15 <= loss_med <= 0.35 and (g_med is None or g_med < 1.0):
        return "健康"
    return "观察"


def crossed_grid(
    previous_step: int,
    current_step: int,
    *,
    start: int,
    stop: int,
    every: int,
    skip: tuple[int, ...] = (),
) -> list[int]:
    if current_step < start:
        return []
    first = ((max(start, previous_step + 1) + every - 1) // every) * every
    found: list[int] = []
    step = first
    skipped = set(skip)
    while step <= current_step and step <= stop:
        if step not in skipped:
            found.append(step)
        step += every
    return found


def gradient_health_report(records: list[dict[str, Any]], step: int) -> str:
    current = slice_window(records, step, HEALTH_10K_EVERY)
    previous = slice_window(records, step - HEALTH_10K_EVERY, HEALTH_10K_EVERY)
    cur = summarize_window(current)
    prev = summarize_window(previous) if previous else None
    status = health_status(cur)
    latest = current[-1] if current else {}
    notes = [
        f"健康状态: {status}",
        f"step: {step}",
        f"scheduled_Kmax: {scheduled_kmax(step):.2f}",
        f"lr: {latest.get('optimizer/learning_rate')}",
        f"sps: {latest.get('system/optimizer_steps_per_second')}",
        f"window: last {HEALTH_10K_EVERY} steps",
        f"loss mean/median/max: {fmt_stat(cur['loss_mean'])} / {fmt_stat(cur['loss_median'])} / {fmt_stat(cur['loss_max'])}",
        f"gnorm mean/median/max: {fmt_stat(cur['gnorm_mean'])} / {fmt_stat(cur['gnorm_median'])} / {fmt_stat(cur['gnorm_max'])}",
        f"clip_fraction max: {fmt_stat(cur['clip_max'])}",
        f"skip_fraction max: {fmt_stat(cur['skip_max'])}",
    ]
    if prev and prev["gnorm_mean"] is not None and cur["gnorm_mean"] is not None:
        delta = float(cur["gnorm_mean"]) - float(prev["gnorm_mean"])
        notes.append(
            f"vs previous 10k gnorm_mean: {fmt_stat(prev['gnorm_mean'])} -> "
            f"{fmt_stat(cur['gnorm_mean'])} (delta {delta:+.4f})"
        )
    notes.extend(
        [
            "",
            "规则：健康=loss 0.15–0.35 且 ‖g‖中位<1；"
            "观察=loss≥0.4 或 ‖g‖中位≥1 / 最大≥5 或 clip≥0.1；"
            "异常=loss≥1 或 ‖g‖≥8 或 clip≥0.5（与即时告警同一套炸法）。",
        ]
    )
    return "\n".join(notes)


def training_window_report(records: list[dict[str, Any]], milestone: int) -> str:
    window = slice_window(records, milestone, 2_000)
    if not window:
        window = records[-12:]
    stats = summarize_window(window)
    latest = window[-1] if window else {}
    return "\n".join(
        [
            f"milestone: {milestone}",
            f"scheduled_Kmax: {scheduled_kmax(milestone):.2f}",
            f"latest_step: {latest.get('global_step')}",
            f"lr: {latest.get('optimizer/learning_rate')}",
            f"loss mean/median/max: {fmt_stat(stats['loss_mean'])} / {fmt_stat(stats['loss_median'])} / {fmt_stat(stats['loss_max'])}",
            f"gnorm mean/median/max: {fmt_stat(stats['gnorm_mean'])} / {fmt_stat(stats['gnorm_median'])} / {fmt_stat(stats['gnorm_max'])}",
            f"clip_fraction max: {fmt_stat(stats['clip_max'])}",
            f"skip_fraction max: {fmt_stat(stats['skip_max'])}",
            f"sps: {latest.get('system/optimizer_steps_per_second')}",
        ]
    )


def deterministic_eval_analysis(
    *,
    milestone: int,
    eval_text: str,
    milestones: tuple[int, ...] | None = None,
    baseline: int | None = None,
) -> str:
    prior = previous_eval_step(milestone, milestones=milestones, baseline=baseline)
    notes = [
        "Phase 2 测评口径（stratified 10%，与此前每 100k 同一协议）：",
        f"- 本档 {milestone}，对照上一档 {prior}。",
        "- 主看约束：全身 / 末端 / 2D root，目标往官方约 4cm 收（600k 全身约 8.9cm）。",
        "- 足滑（Skate）看有没有进步。",
        "- FID / R@3 / 接触只判断相对上一档有没有退步，不要求再创新高。",
    ]
    return "\n".join(notes) + "\n\n" + eval_text


EVAL_MIMO_SYSTEM = (
    "你是 Kimodo V2 1M Phase 2 评测助手。用中文、短段落。"
    "主看约束（全身/末端/2D root）有没有往官方约 4cm 收。"
    "FID、R@3、接触只判断相对上一档有没有退步。"
    "足滑看有没有进步。不要编造没有出现的数字。"
)
HEAD_TO_HEAD_SYSTEM = (
    "你是 Kimodo 评测对照助手。用中文写全面对比分析，分短段落。"
    "对象是：官方 SEED-v1.1 vs 我们 695k 健康档，同一套 stratified 10% / paper protocol。"
    "必须覆盖 content 和 repetition：全身/末端/2D root 约束、足滑、FID、R@3、接触。"
    "对照官方约 4cm 约束。不要编造未出现的数字。"
    "若文中有 600k/650k，只作为轨迹补充，不要喧宾夺主。"
    "最后给一句明确结论：695k 相对官方是接近、部分接近还是明显落后，"
    "以及值不值得在 695k 继续 fork。"
)
HEALTH_MIMO_SYSTEM = (
    "你是 Kimodo 训练健康助手。用中文、短段落。"
    "根据 loss 和梯度判断健康/观察/异常是否合理，是否像 696.8k 那种炸法。"
    "不要编造测评数字。"
)
DEFAULT_MIMO_BASE = "https://api.xiaomimimo.com/v1"
DEFAULT_MIMO_MODEL = "mimo-v2.5-pro"


def llm_analysis(prompt: str, *, system: str = EVAL_MIMO_SYSTEM, attempts: int = 3) -> str | None:
    watch.load_env_files()
    key = os.environ.get("PRODUCT_GRAPH_LLM_API_KEY", "").strip()
    base = (os.environ.get("PRODUCT_GRAPH_LLM_BASE_URL", "") or DEFAULT_MIMO_BASE).strip().rstrip("/")
    model = (os.environ.get("PRODUCT_GRAPH_LLM_MODEL", "") or DEFAULT_MIMO_MODEL).strip()
    if not key:
        watch.log("MiMo skipped: PRODUCT_GRAPH_LLM_API_KEY empty")
        return None
    payload = json.dumps(
        {
            "model": model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        },
        ensure_ascii=False,
    ).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(
            f"{base}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
            text = str(body["choices"][0]["message"]["content"]).strip()
            if text:
                return text
            last_error = RuntimeError("empty MiMo content")
        except (
            urllib.error.URLError,
            KeyError,
            IndexError,
            TimeoutError,
            json.JSONDecodeError,
            RuntimeError,
        ) as error:
            last_error = error
            watch.log(f"MiMo analysis failed attempt {attempt}/{attempts}: {error}")
            if attempt < attempts:
                time.sleep(2)
    watch.log(f"MiMo analysis gave up: {last_error}")
    return None


def read_jsonl_tail(path: Path, limit: int = 400) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    rows: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def eval_summary_path(eval_root: Path, step: int) -> Path:
    return eval_root / f"step-{step:09d}" / "summary_rows.json"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


def _env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def previous_eval_step(
    milestone: int,
    *,
    milestones: tuple[int, ...] | None = None,
    baseline: int | None = None,
) -> int:
    miles = MILESTONES if milestones is None else milestones
    base = FORK_BASELINE_STEP if baseline is None else baseline
    if milestone in miles:
        index = miles.index(milestone)
        if index > 0:
            return miles[index - 1]
    return base


def load_eval_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or "tables" not in payload:
        return None
    return payload


def compose_head_to_head_text(
    *,
    official: dict[str, Any],
    fork: dict[str, Any],
    extra: list[tuple[str, dict[str, Any]]],
) -> str:
    chunks = [
        "全面对照：Official SEED-v1.1 vs 695k（同一 stratified 10%，paper protocol，100 steps）。",
        "官方窗口 17:30–17:50；695k 窗口 17:45–18:05。",
        "主看约束距离官方约 4cm；足滑 / FID / R@3 / 接触一并对照。",
        "compare 方向：Official -> 695k。进步表示 695k 优于官方。",
    ]
    for split, title in (("content", "content（主表）"), ("repetition", "repetition")):
        official_h = extract_eval_highlights(official, split=split)
        fork_h = extract_eval_highlights(fork, split=split)
        chunks.append(f"\n## {title}")
        chunks.append(
            compare_eval_highlights(
                fork_h,
                official_h,
                prior_step=0,
                milestone=695_000,
                title=f"{title}: Official SEED-v1.1 -> 695k",
            )
        )
        chunks.append(format_highlights("Official SEED-v1.1", official_h))
        chunks.append(format_highlights("695k kf-smooth", fork_h))
        for label, summary in extra:
            chunks.append(
                format_highlights(label, extract_eval_highlights(summary, split=split))
            )
    return "\n".join(chunks)


def maybe_email_official_vs_695k(
    *,
    state: dict[str, Any],
    official_path: Path,
    fork_path: Path,
    extra_paths: tuple[tuple[str, str], ...],
    run_dir: Path,
    to_addr: str,
    from_addr: str,
    now: float,
) -> bool:
    if state.get("emailed_official_vs_695k"):
        return False
    official = load_eval_summary(official_path)
    fork = load_eval_summary(fork_path)
    if official is None or fork is None:
        last = float(state.get("paired_wait_log_unix") or 0)
        if now - last >= 600:
            state["paired_wait_log_unix"] = now
            watch.log(
                "waiting official_v1="
                f"{'ready' if official else 'pending'} "
                f"695k={'ready' if fork else 'pending'}"
            )
        return False
    extra: list[tuple[str, dict[str, Any]]] = []
    for label, raw in extra_paths:
        summary = load_eval_summary(Path(raw))
        if summary is not None:
            extra.append((label, summary))
    eval_text = compose_head_to_head_text(
        official=official, fork=fork, extra=extra
    )
    llm = llm_analysis(
        "下面是 Official SEED-v1.1 与 695k 的同一协议对照。"
        "请做全面对比分析，不要讨论 train loss。\n\n" + eval_text,
        system=HEAD_TO_HEAD_SYSTEM,
    )
    if not llm:
        watch.log("defer official-vs-695k: waiting for MiMo analysis")
        return False
    sent = send_report(
        subject_kind="official-vs-695k",
        alerts=[watch.Alert("eval_ready", "Official SEED-v1.1 vs 695k head-to-head")],
        body=eval_text + "\n\nMiMo 全面对比分析：\n" + llm,
        run_dir=run_dir,
        to_addr=to_addr,
        from_addr=from_addr,
        record=None,
    )
    if sent:
        state["emailed_official_vs_695k"] = True
        watch.log("emailed official-vs-695k head-to-head")
    return sent


def compose_eval_text(
    eval_root: Path,
    prior_root: Path,
    milestone: int,
    *,
    milestones: tuple[int, ...] | None = None,
    baseline: int | None = None,
) -> str:
    current = eval_summary_path(eval_root, milestone)
    if not current.is_file():
        return (
            f"测评尚未就绪: {current}\n"
            "开发机无 GPU，不会在本 pod 跑 benchmark。另开 1xH200 评测后把 "
            "summary_rows.json 写到上述路径，守护进程会补发。"
        )
    summary = json.loads(current.read_text(encoding="utf-8"))
    current_h = extract_eval_highlights(summary)
    chunks = [format_highlights(f"current eval {milestone}", current_h)]
    prior_step = previous_eval_step(
        milestone, milestones=milestones, baseline=baseline
    )
    prior = eval_summary_path(eval_root, prior_step)
    if not prior.is_file():
        prior = eval_summary_path(prior_root, prior_step)
    if prior.is_file():
        prior_h = extract_eval_highlights(json.loads(prior.read_text(encoding="utf-8")))
        chunks.append(
            format_highlights(
                f"previous {prior_step} ({prior.parent.parent.name})",
                prior_h,
            )
        )
        chunks.insert(0, compare_eval_highlights(
            current_h, prior_h, prior_step=prior_step, milestone=milestone
        ))
    return "\n".join(chunks)


def send_report(
    *,
    subject_kind: str,
    alerts: list[watch.Alert],
    body: str,
    run_dir: Path,
    to_addr: str,
    from_addr: str,
    record: dict[str, Any] | None,
) -> bool:
    watch.load_env_files()
    if not os.environ.get("KIMODO_ALERT_SMTP_PASSWORD"):
        watch.log(f"defer {subject_kind}: SMTP password still empty")
        return False
    message = watch.build_email(
        to_addr=to_addr,
        from_addr=from_addr,
        run_dir=run_dir,
        alerts=alerts,
        record=record,
        extra=body,
    )
    # build_email overwrites Subject; patch in the kind.
    message.replace_header("Subject", f"[kimodo] {subject_kind} {message['Subject']}")
    try:
        watch.send_email(message)
    except Exception as error:  # noqa: BLE001 - keep watching after SMTP blips
        watch.log(f"SMTP send failed ({subject_kind}): {error}")
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(os.environ.get("KIMODO_RUN_DIR", watch.DEFAULT_RUN_DIR)),
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path(os.environ.get("KIMODO_EVAL_ROOT", DEFAULT_EVAL_ROOT)),
    )
    parser.add_argument(
        "--prior-eval-root",
        type=Path,
        default=Path(os.environ.get("KIMODO_PRIOR_EVAL_ROOT", DEFAULT_PRIOR_EVAL_ROOT)),
    )
    parser.add_argument(
        "--official-eval-summary",
        type=Path,
        default=Path(
            os.environ.get("KIMODO_OFFICIAL_EVAL_SUMMARY", DEFAULT_OFFICIAL_EVAL_SUMMARY)
        ),
    )
    parser.add_argument(
        "--fork-695k-eval-summary",
        type=Path,
        default=Path(os.environ.get("KIMODO_695K_EVAL_SUMMARY", DEFAULT_695K_EVAL_SUMMARY)),
    )
    parser.add_argument(
        "--watch-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "KIMODO_WATCH_DIR",
                str(Path(os.environ.get("KIMODO_CODE_ROOT", "/home/share/yzt/kimodo-reproduction")) / "watch" / "v2-1m-hostnet-cap800-from695k"),
            )
        ),
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--cooldown-seconds", type=float, default=1200.0)
    parser.add_argument("--stall-seconds", type=float, default=720.0)
    parser.add_argument("--to", default=os.environ.get("KIMODO_ALERT_EMAIL", watch.DEFAULT_TO))
    parser.add_argument(
        "--from-addr",
        default=os.environ.get("KIMODO_ALERT_SMTP_USER", watch.DEFAULT_TO),
    )
    parser.add_argument(
        "--health-start",
        type=int,
        default=_env_int("KIMODO_HEALTH_10K_START", HEALTH_10K_START),
    )
    parser.add_argument(
        "--health-every",
        type=int,
        default=_env_int("KIMODO_HEALTH_10K_EVERY", HEALTH_10K_EVERY),
    )
    parser.add_argument(
        "--eval-start",
        type=int,
        default=_env_int("KIMODO_EVAL_MILESTONE_START", MILESTONES[0]),
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=_env_int("KIMODO_EVAL_MILESTONE_EVERY", 50_000),
    )
    parser.add_argument(
        "--eval-stop",
        type=int,
        default=_env_int("KIMODO_EVAL_MILESTONE_STOP", MILESTONES[-1]),
    )
    parser.add_argument(
        "--fork-baseline",
        type=int,
        default=_env_int("KIMODO_FORK_BASELINE_STEP", FORK_BASELINE_STEP),
    )
    parser.add_argument(
        "--head-to-head",
        dest="head_to_head",
        action="store_true",
        default=_env_flag("KIMODO_HEAD_TO_HEAD", "1"),
    )
    parser.add_argument(
        "--no-head-to-head",
        dest="head_to_head",
        action="store_false",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    jsonl = run_dir / "train.jsonl"
    watch_dir = args.watch_dir.expanduser().resolve()
    state_path = watch_dir / "state.json"
    watch_dir.mkdir(parents=True, exist_ok=True)
    watch.load_env_files()
    milestones = tuple(
        range(args.eval_start, args.eval_stop + 1, max(1, args.eval_every))
    )
    watch.log(f"daemon watching {jsonl}")
    watch.log(f"eval root {args.eval_root} mail={args.to}")
    watch.log(
        f"health every {args.health_every} from {args.health_start}; "
        f"eval {list(milestones[:3])}... baseline={args.fork_baseline}"
    )
    if args.head_to_head:
        watch.log(
            f"head-to-head official={args.official_eval_summary} "
            f"695k={args.fork_695k_eval_summary}"
        )
    else:
        watch.log("head-to-head disabled")
    state = watch.load_state(state_path)
    state.setdefault("emailed_milestones", [])
    state.setdefault("emailed_evals", [])
    state.setdefault("emailed_10k", [])
    state.setdefault("startup_emailed", False)

    while True:
        watch.load_env_files()
        now = time.time()
        previous_step = int(state.get("last_step") or 0)
        events, state = watch.watch_once(
            jsonl=jsonl,
            state=state,
            stall_s=args.stall_seconds,
            now=now,
        )
        current_step = int(state.get("last_step") or previous_step)

        if not state.get("startup_emailed"):
            last_try = float(state.get("startup_last_attempt_unix") or 0)
            if now - last_try >= 600:
                state["startup_last_attempt_unix"] = now
                latest = read_jsonl_tail(jsonl, 3)
                record = latest[-1] if latest else None
                body = (
                    "开发机守护已启动。电脑可以带走。\n"
                f"当前 step={current_step}。loss 异常会立即发信；"
                f"每 {args.health_every // 1000}k 发梯度健康；"
                f"{args.eval_start // 1000}k 起每 {args.eval_every // 1000}k 只发 benchmark 指标分析。\n"
                    "开发机无 GPU，benchmark 需另开 1xH200。"
                )
                sent = send_report(
                    subject_kind="watcher-online",
                    alerts=[watch.Alert("startup", f"monitor online step={current_step}")],
                    body=body,
                    run_dir=run_dir,
                    to_addr=args.to,
                    from_addr=args.from_addr,
                    record=record,
                )
                if sent:
                    state["startup_emailed"] = True
                    watch.log("startup email sent")

        for alerts, record in events:
            due = watch.maybe_alert(
                alerts=alerts,
                state=state,
                cooldown_s=args.cooldown_seconds,
                now=now,
            )
            if not due:
                continue
            watch.log("; ".join(alert.summary for alert in due))
            sent = send_report(
                subject_kind="LOSS-ALERT",
                alerts=due,
                body="对照上次 696.8k 坍缩：若 loss≥1 或 clip 打满，优先停任务而不是继续训。",
                run_dir=run_dir,
                to_addr=args.to,
                from_addr=args.from_addr,
                record=record,
            )
            if not sent:
                for alert in due:
                    state.setdefault("last_alert_unix", {}).pop(alert.reason, None)

        emailed_10k = [int(value) for value in state.get("emailed_10k", [])]
        due_10k = [
            step
            for step in crossed_grid(
                previous_step,
                current_step,
                start=args.health_start,
                stop=args.eval_stop,
                every=args.health_every,
                skip=(),
            )
            if step not in emailed_10k
        ]
        if not due_10k and not emailed_10k and current_step >= args.health_start:
            aligned = current_step - (current_step % args.health_every)
            if aligned >= args.health_start:
                due_10k = [aligned]
        if due_10k:
            rows = read_jsonl_tail(jsonl, 2_500)
            for step in due_10k:
                report = gradient_health_report(rows, step)
                llm = llm_analysis(
                    "这是 10k 梯度健康快照，不要写测评数字。"
                    "判断健康/观察/异常是否合理，梯度是否在爬向 696.8k 那种炸法。\n\n"
                    + report,
                    system=HEALTH_MIMO_SYSTEM,
                )
                body = report + (
                    "\n\nMiMo 分析：\n" + llm if llm else "\n\nMiMo 分析：本次未拿到，仅规则判断。"
                )
                latest = rows[-1] if rows else None
                sent = send_report(
                    subject_kind=f"{step // 1000}k-health",
                    alerts=[watch.Alert("health_10k", f"10k health snapshot step={step}")],
                    body=body,
                    run_dir=run_dir,
                    to_addr=args.to,
                    from_addr=args.from_addr,
                    record=latest,
                )
                if sent:
                    emailed_10k.append(step)
                    watch.log(f"emailed {step} 10k health")
        state["emailed_10k"] = emailed_10k

        emailed_evals = [int(value) for value in state.get("emailed_evals", [])]
        for milestone in milestones:
            if milestone in emailed_evals:
                continue
            summary = eval_summary_path(args.eval_root, milestone)
            if not summary.is_file():
                continue
            eval_text = compose_eval_text(
                args.eval_root,
                args.prior_eval_root,
                milestone,
                milestones=milestones,
                baseline=args.fork_baseline,
            )
            rule = deterministic_eval_analysis(
                milestone=milestone,
                eval_text=eval_text,
                milestones=milestones,
                baseline=args.fork_baseline,
            )
            llm = llm_analysis(
                "下面是 Phase 2 的 stratified 10% 测评。"
                "主写约束有没有收；先验指标只说有没有退步；足滑只说有没有进步。"
                "不要讨论 train loss。\n\n" + rule,
                system=EVAL_MIMO_SYSTEM,
            )
            if not llm:
                watch.log(f"defer {milestone} eval mail: waiting for MiMo analysis")
                continue
            body = rule + "\n\nMiMo 分析：\n" + llm
            sent = send_report(
                subject_kind=f"{milestone // 1000}k-eval",
                alerts=[watch.Alert("eval_ready", f"eval summary for {milestone}")],
                body=body,
                run_dir=run_dir,
                to_addr=args.to,
                from_addr=args.from_addr,
                record=None,
            )
            if sent:
                emailed_evals.append(milestone)
                watch.log(f"emailed {milestone} eval")
        state["emailed_evals"] = emailed_evals
        if args.head_to_head:
            maybe_email_official_vs_695k(
                state=state,
                official_path=args.official_eval_summary.expanduser(),
                fork_path=args.fork_695k_eval_summary.expanduser(),
                extra_paths=CONTEXT_EVAL_SUMMARIES,
                run_dir=run_dir,
                to_addr=args.to,
                from_addr=args.from_addr,
                now=now,
            )
        watch.save_state(state_path, state)
        time.sleep(max(5.0, args.poll_seconds))


if __name__ == "__main__":
    sys.exit(main())
