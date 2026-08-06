"""Generate renderer-portable, presentation-grade SVG flow charts.

The output deliberately uses only SVG rect/path/text/tspan primitives.  It does
not use foreignObject, so the diagrams remain visible in browsers, IDE Markdown
previews, ImageMagick, and common PDF exporters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

try:
    from PIL import ImageFont
except ImportError:  # pragma: no cover - only affects optional diagram regeneration
    ImageFont = None


WIDTH = 3600
MARGIN = 120
CONTENT_W = WIDTH - 2 * MARGIN
PANEL_GAP = 28
# Put the installed CJK font first.  Some SVG/PDF renderers do not perform
# per-glyph fallback after selecting Arial, which otherwise makes Chinese
# characters disappear while Latin text remains visible.
FONT = "DejaVu Sans"
MONO = "DejaVu Sans Mono"
CJK_FONT = "Droid Sans Fallback"

COLORS = {
    "source": ("#EAF2FF", "#3973C6", "SOURCE / PUBLIC ARTIFACT"),
    "paper": ("#E8F7EE", "#24815A", "PAPER"),
    "code": ("#EEF5EE", "#5C8465", "NVIDIA CODE / CONFIG"),
    "recon": ("#FFF1D6", "#D97706", "RECON · 本仓实现"),
    "product": ("#F1ECFF", "#7457B5", "PRODUCT / TENSOR"),
    "boundary": ("#FFE8E6", "#C24132", "BOUNDARY / BLOCKED"),
}


@dataclass(frozen=True)
class Panel:
    title: str
    kind: str
    lines: tuple[str, ...]


@dataclass(frozen=True)
class Stage:
    title: str
    panels: tuple[Panel, ...]
    edge: str = ""
    note: str = ""


def _text(x: float, y: float, value: str, size: int, weight: int = 400,
          color: str = "#172033", anchor: str = "middle", family: str = FONT) -> str:
    # librsvg treats font-changing tspans as separate text-anchor chunks.  That
    # makes every Latin/CJK run start at the same center and visibly overlap.
    # Measure both installed fonts and position the runs explicitly instead.
    runs: list[tuple[bool, str]] = []
    for char in value:
        codepoint = ord(char)
        is_cjk = (
            0x2E80 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
            or 0xFF00 <= codepoint <= 0xFFEF
        )
        if runs and runs[-1][0] == is_cjk:
            runs[-1] = (is_cjk, runs[-1][1] + char)
        else:
            runs.append((is_cjk, char))
    widths: list[float] = []
    for is_cjk, run in runs:
        if ImageFont is not None:
            if is_cjk:
                font_path = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
            elif family == MONO:
                suffix = "-Bold" if weight >= 700 else ""
                font_path = f"/usr/share/fonts/truetype/dejavu/DejaVuSansMono{suffix}.ttf"
            else:
                suffix = "-Bold" if weight >= 700 else ""
                font_path = f"/usr/share/fonts/truetype/dejavu/DejaVuSans{suffix}.ttf"
            widths.append(float(ImageFont.truetype(font_path, size).getlength(run)))
        else:
            widths.append(size * sum(1.0 if is_cjk else (0.64 if ch.isalnum() else 0.42) for ch in run))

    if anchor == "middle":
        cursor = x - sum(widths) / 2
    elif anchor == "end":
        cursor = x - sum(widths)
    else:
        cursor = x
    spans: list[str] = []
    for (is_cjk, run), run_width in zip(runs, widths, strict=True):
        spans.append(
            f'<tspan x="{cursor:.1f}" font-family="{CJK_FONT if is_cjk else family}">{escape(run)}</tspan>'
        )
        cursor += run_width
    content = "".join(spans)
    return (
        f'<text x="{x:.0f}" y="{y:.0f}" text-anchor="start" '
        f'font-family="{family}" font-size="{size}" font-weight="{weight}" '
        f'fill="{color}">{content}</text>'
    )


def _panel_height(panel: Panel) -> int:
    return 135 + len(panel.lines) * 43


def render(title: str, subtitle: str, stages: tuple[Stage, ...], output: Path,
           footer: str = "") -> None:
    stage_heights = []
    for stage in stages:
        content_h = max(_panel_height(p) for p in stage.panels)
        stage_heights.append(130 + content_h + (70 if stage.note else 24))
    header_h = 350
    edge_h = 125
    footer_h = 180 if footer else 70
    height = 70 + header_h + sum(stage_heights) + edge_h * (len(stages) - 1) + footer_h

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}" role="img">',
        f'<title>{escape(title)}</title>',
        f'<desc>{escape(subtitle)}</desc>',
        f'<rect width="{WIDTH}" height="{height}" fill="#F8FAFC"/>',
        '<defs><marker id="arrow" markerWidth="20" markerHeight="20" viewBox="0 0 20 20" refX="17" refY="10" orient="auto" markerUnits="userSpaceOnUse"><path d="M1,1 L18,10 L1,19 z" fill="#475569"/></marker></defs>',
        _text(WIDTH / 2, 130, title, 58, 900, "#0F172A"),
        _text(WIDTH / 2, 190, subtitle, 29, 400, "#334155"),
    ]
    legend = [
        ("PAPER", "paper"), ("NVIDIA CODE / CONFIG", "code"),
        ("RECON · 本仓实现", "recon"), ("PRODUCT / TENSOR", "product"),
        ("BOUNDARY / BLOCKED", "boundary"),
    ]
    lx = 390
    for label, kind in legend:
        fill, stroke, _ = COLORS[kind]
        w = 500 if kind != "boundary" else 560
        out.append(f'<rect x="{lx}" y="235" width="{w}" height="58" rx="29" fill="{fill}" stroke="{stroke}" stroke-width="3"/>')
        out.append(_text(lx + w / 2, 274, label, 22, 800, stroke))
        lx += w + 28

    y = 70 + header_h
    for index, (stage, stage_h) in enumerate(zip(stages, stage_heights, strict=True)):
        out.append(f'<rect x="{MARGIN}" y="{y}" width="{CONTENT_W}" height="{stage_h}" rx="28" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="4"/>')
        out.append(_text(WIDTH / 2, y + 65, stage.title, 38, 850, "#0F172A"))
        n = len(stage.panels)
        panel_w = (CONTENT_W - 80 - PANEL_GAP * (n - 1)) / n
        panel_y = y + 100
        panel_h = stage_h - 125 - (65 if stage.note else 0)
        for pidx, panel in enumerate(stage.panels):
            px = MARGIN + 40 + pidx * (panel_w + PANEL_GAP)
            fill, stroke, evidence = COLORS[panel.kind]
            dash = ' stroke-dasharray="16 12"' if panel.kind == "boundary" else ""
            out.append(f'<rect x="{px:.0f}" y="{panel_y}" width="{panel_w:.0f}" height="{panel_h}" rx="22" fill="{fill}" stroke="{stroke}" stroke-width="4"{dash}/>')
            out.append(_text(px + panel_w / 2, panel_y + 45, evidence, 19, 850, stroke))
            out.append(_text(px + panel_w / 2, panel_y + 91, panel.title, 30, 850, "#111827"))
            line_y = panel_y + 137
            for line in panel.lines:
                family = MONO if line.startswith("⟦") else FONT
                shown = line[1:-1] if line.startswith("⟦") and line.endswith("⟧") else line
                out.append(_text(px + panel_w / 2, line_y, shown, 25, 450, "#172033", family=family))
                line_y += 43
        if stage.note:
            out.append(_text(WIDTH / 2, y + stage_h - 27, stage.note, 24, 650, "#475569"))
        y += stage_h
        if index < len(stages) - 1:
            out.append(f'<path d="M{WIDTH/2:.0f} {y + 10} L{WIDTH/2:.0f} {y + 92}" stroke="#475569" stroke-width="7" fill="none"/>')
            if stage.edge:
                pill_w = min(1800, max(640, len(stage.edge) * 29))
                out.append(f'<rect x="{WIDTH/2-pill_w/2:.0f}" y="{y + 25}" width="{pill_w}" height="46" rx="23" fill="#F8FAFC"/>')
                out.append(_text(WIDTH / 2, y + 57, stage.edge, 22, 700, "#475569"))
            out.append(f'<polygon points="{WIDTH/2-14:.0f},{y+83} {WIDTH/2+14:.0f},{y+83} {WIDTH/2:.0f},{y+107}" fill="#475569"/>')
            y += edge_h
    if footer:
        out.append(f'<rect x="{MARGIN}" y="{y + 35}" width="{CONTENT_W}" height="88" rx="22" fill="#FFE8E6" stroke="#C24132" stroke-width="3" stroke-dasharray="14 10"/>')
        out.append(_text(WIDTH / 2, y + 91, footer, 24, 750, "#7F1D1D"))
    out.append("</svg>")
    output.write_text("\n".join(out) + "\n", encoding="utf-8")


DATA_STAGES = (
    Stage("阶段 D0｜公开输入：动作、语义、split、文本编码器是四类独立资产", (
        Panel("BONES-SEED / SOMA-Uniform BVH", "source", (
            "每个 BVH = 一段连续录制，不等于一个语义 event", "HIERARCHY：SOMA77 父子关系 + rest offsets",
            "MOTION：root XYZ + 77 关节 local Euler", "源采样率 120 FPS；root 位移单位 cm",
        )),
        Panel("CSV + temporal JSONL + train split", "source", (
            "CSV：filename / move_soma_uniform_path / 7 个 content_*", "content_*：同一 motion 的整段自然语言描述",
            "temporal JSONL：events[start_time,end_time,description]", "split：允许进入训练的 motion key；防止集合泄漏",
        )),
        Panel("LLM2Vec 离线编码器资产", "code", (
            "PAPER：LLM2Vec 4096D 作为文本条件", "CODE：Llama-3 8B compatibility foundation",
            "CODE：MNTP + supervised LLM2Vec adapters", "只离线编码句向量；动作训练不加载 8B 模型",
        )),
    ), edge="固定 revision / size / SHA-256；tar 安全解压；按官方 split 选择 motion"),
    Stage("阶段 D1｜资源校验与训练 motion 集合选择", (
        Panel("选择输入", "source", (
            "CSV.move_soma_uniform_path", "官方 train_split_paths.txt", "安全解压后的 SOMA-Uniform BVH 树",
        )),
        Panel("本仓选择协议", "recon", (
            "拒绝 tar 绝对路径、..、链接和特殊文件", "路径统一成 canonical motion key 后求交集",
            "selected = CSV path ∩ official train split", "固定缺失集合，不对 36 个缺失 key 静默猜测替代品",
        )),
        Panel("确定的动作集合", "product", (
            "official split keys：128,351", "metadata + BVH 可匹配：128,315", "每个 key 可追溯到 BVH、CSV row、temporal events",
        )),
    ), edge="对每个选中 BVH 执行单位、旋转参考、帧率与骨架标准化"),
    Stage("阶段 D2A｜公开 BVH primitive：解析通道、统一单位、转换旋转参考", (
        Panel("BVH 原始量", "source", (
            "root translation [Tsrc,3]：世界坐标，cm，Y-up", "local Euler [Tsrc,77,3]：关节相对父关节，degree",
            "每关节 CHANNEL 顺序决定 Euler 乘法顺序", "Frame Time≈1/120 s；rest offsets 定义静止骨架",
        )),
        Panel("公开 loader / skeleton 操作", "code", (
            "CODE：按原生 Euler order → 3×3 rotation matrix", "CODE：root ×0.01，cm→m；BVH rest pose→T-pose",
            "T-pose 只改变关节旋转参考基", "不改 root 世界轨迹；也不是生成新的动作内容",
        )),
        Panel("公开 primitive 的中间量", "product", (
            "local rotations [Tsrc,77,3,3]", "含义：T-pose 下、每关节相对父关节的旋转",
            "root positions [Tsrc,3]：Y-up、m", "fps 来自 BVH Frame Time；这里仍约为 120",
        )),
    ), edge="本仓固定 BONES 时间合同与训练 skeleton 合同，并执行强校验"),
    Stage("阶段 D2B｜本仓批量标准化：120→30、SOMA77→SOMA30、验证并原子写 NPZ", (
        Panel("公开 primitive 中间量", "source", (
            "local rotations [Tsrc,77,3,3]", "root positions [Tsrc,3]，单位 m", "source FPS 必须严格符合 BONES 120 FPS 合同",
        )),
        Panel("本仓 converter policy", "recon", (
            "严格 [::4]，120→30 FPS；不做插值", "按固定关节名选择 SOMA30；不是 IK retarget",
            "检查 shape / finite / 最少 2 帧 / 帧数一致", "检查 RᵀR≈I、det(R)≈1；记录 source SHA",
            "临时文件完成后 os.replace，避免暴露半写 NPZ",
        )),
        Panel("canonical NPZ：稳定基础运动", "product", (
            "local_rot_mats float32 [T,30,3,3]", "含义：T-pose 参考下，每关节相对父关节的旋转",
            "root_positions float32 [T,3]：pelvis 世界 XYZ，m", "fps float32 scalar=30：定义时间与速度单位",
            "source_provenance_json：源 SHA / converter / 采样参数", "provenance 只审计；不作为神经网络输入",
        )),
    ), edge="NPZ 只保留可复用基础运动；随后由 annotation 创建“时间范围 ↔ 文本”训练行",
    note="为什么不是 369D：369D 依赖 manifest row 的时间范围、每 epoch crop 与随机 heading；提前保存会复制数据并冻结在线增强。"),
    Stage("阶段 D3｜annotation → raw manifest：motion、event、caption、row 不是一回事", (
        Panel("一份 motion 上的语义层", "source", (
            "motion：一段连续人体运动 NPZ", "event：motion 内一个有 start/end 的语义时间段",
            "caption：描述整段 motion 或某个 event 的一句文字", "同一 motion 可对应 7 个 full captions + 多个 events",
        )),
        Panel("本仓 manifest builder", "recon", (
            "full row：整段 NPZ + 一个 CSV content_*", "event row：同一 NPZ + start/end + temporal description",
            "combined row：同一 motion 相邻 events 总区间 + Then 文本", "combined 只组合索引与文字；没有生成任何新 BVH/NPZ",
        )),
        Panel("train.raw.jsonl", "product", (
            "id / motion / text / optional start_time,end_time", "frame_count / source_fps / sample_kind / text_source",
            "augmentation_provenance / split", "full 898,205；event 318,647；combined 190,332",
            "总计 1,407,184 rows，共享 128,315 motion NPZ",
        )),
    ), edge="对 raw rows 的唯一 sanitized text 进行一次离线编码并缓存"),
    Stage("阶段 D4｜raw text → LLM2Vec cache → cached manifest", (
        Panel("编码输入", "source", (
            "train.raw.jsonl 中的自然语言 text", "strip / 合并空白得到 sanitized text", "同一 motion 的不同 captions 保持为不同训练 rows",
        )),
        Panel("编码与缓存协议", "recon", (
            "PAPER：使用 LLM2Vec 生成文本条件", "cache_key = SHA256(encoder identity + NUL + text)",
            "唯一 text 只编码一次；mean-pool 得 [1,4096]", "sidecar 固定 text hash / encoder hash / shape / file SHA",
        )),
        Panel("缓存产物", "product", (
            "text-cache/key.npy float32 [1,4096]", "key.npy.metadata.json：内容与编码器身份",
            "train.cached.jsonl：raw row + embedding path/key/hash", "132,972 unique embeddings 被 1,407,184 rows 复用",
        )),
    ), edge="扫描训练 spans，在统一物理表示上拟合固定逐维 mean/std"),
    Stage("阶段 D5｜stats 与 self-contained bundle：建立固定数值空间合同", (
        Panel("stats 从哪里来", "source", (
            "cached manifest 决定唯一 motion/time spans", "NPZ → FK → 物理单位 369D；最长 300 帧分窗",
            "执行与训练一致的首帧 XZ 归零 + 确定性 heading", "stats 不使用 embedding 数值，也不是每 batch 重算",
        )),
        Panel("本仓拟合 recipe", "recon", (
            "跨全部训练帧，按每个 feature dimension 累计", "float64 sum / sum-square → float32 mean/std",
            "由真值 global5 派生 local4，只为拟合 bridge 条件尺度", "global root [5] / local root [4] / body [364]",
        )),
        Panel("bundle 产物", "product", (
            "stats/*/mean.npy + std.npy", "motions/ + text-cache/ + train.cached.jsonl",
            "relative path inventory：path / size / SHA-256", "resource-state.json + repro.paths.yaml",
            "不逐 motion 保存 local4；只保存其四维统计量",
        )),
    ), edge="Dataset 每次 __getitem__ 根据 row 动态切片、做运动学与随机几何变换"),
    Stage("阶段 D6｜ONLINE Dataset：canonical NPZ 在这里才派生成 369D", (
        Panel("取样与坐标标准化", "recon", (
            "row 选择 full/event/combined 的 start/end span", "span>300：按 epoch + index seed 随机 crop",
            "首帧 smooth-root XZ 平移到 (0,0)", "整段绕 Y 轴旋转到抽样 heading；各样本世界原点互不共享",
        )),
        Panel("FK 与特征派生", "code", (
            "FK(local rotations + offsets + root) → global pos/rot", "smooth root XYZ 3；heading cos/sin 2",
            "joint positions 30×3 = 90", "global joint rotations 30×6D = 180",
            "joint velocities：position difference ×30 FPS = 90", "heel/toe contact：高度+速度阈值 = 4",
        )),
        Panel("归一化后的模型运动", "product", (
            "global root：3 + 2 = 5", "body：90 + 180 + 90 + 4 = 364",
            "总计 5 + 364 = 369 features / frame", "global5 与 body364 使用固定全训练集 stats 逐维 normalize",
            "同时加载对应的 [1,4096] 离线文本向量",
        )),
    ), edge="collate：按 batch 内 Tmax padding，并显式给出 valid mask"),
    Stage("阶段 D7｜送入训练器的最终 batch 合同", (
        Panel("Motion tensors", "product", (
            "clean_motion float32 [B,Tmax,369]", "valid_frames bool [B,Tmax]", "lengths int64 [B]",
        )),
        Panel("Text + geometry condition", "product", (
            "text_features float32 [B,1,4096]", "text_pad_mask bool [B,1]", "first_heading_angle float32 [B]",
        )),
        Panel("padding 与复现性", "recon", (
            "padding 仍可能参与批量张量计算", "valid_frames 使 padding 不贡献 loss / denominator",
            "epoch/index seed 固定 crop 与随机 heading", "trainer 不再读取 CSV、temporal JSONL 或 BVH",
        )),
    )),
)


ADOPTION_STAGES = (
    Stage("阶段 A0｜当前输入是已经存在的训练成品，不是本轮 fresh converter 输出", (
        Panel("legacy motions", "source", ("128,315 个 SOMA30 / 30 FPS NPZ", "local_rot_mats [T,30,3,3]", "root_positions [T,3]")),
        Panel("legacy text + manifests", "source", ("132,972 个 LLM2Vec [1,4096]", "embedding metadata sidecars", "raw/cached manifests：共 1,407,184 rows")),
        Panel("legacy stats", "source", ("global-root mean/std [5]", "local-root mean/std [4]", "body mean/std [364]")),
    ), edge="先验证 shape、dtype、hash、provider identity、路径边界与 manifest schema"),
    Stage("阶段 A1｜第一层验证：不满足合同就停止，不发布 train-ready", (
        Panel("Motion 合同", "recon", ("必须包含两个必需数组", "shape=[T,30,3,3] / [T,3]", "float32、finite、帧数一致", "引用不得逃出 legacy root")),
        Panel("Text / index 合同", "recon", ("embedding 必须为 float32 [1,4096]", "cache key / text hash / encoder identity 对齐", "file SHA-256 / frame_count / time range 可核验")),
        Panel("Stats 合同", "recon", ("维数必须为 5 / 4 / 364", "mean/std 必须 finite float32", "std 必须能安全归一化")),
    ), edge="在隐藏 staging 中复用 immutable payload；重写可迁移索引"),
    Stage("阶段 A2｜hardlink 的准确含义：新路径名与旧路径名指向同一 inode", (
        Panel("不是这些操作", "boundary", ("不是重新解析 BVH", "不是重新生成 NPZ", "不是复制第二份 34 GB 数据", "不是 symlink；不依赖旧路径仍存在")),
        Panel("os.link(source, destination)", "recon", ("legacy/A.npz ─┐", "　　　　　├─ same inode ─ same data blocks", "prepared/A.npz ─┘", "必须同 filesystem；payload 按 immutable 只读")),
        Panel("需要重新写的内容", "recon", ("旧绝对路径 → bundle 内相对路径", "补 frame_count / metadata path / content SHA", "保留历史 converter / encoder provenance", "不冒充 fresh reconstruction")),
    ), edge="对新 bundle 做全引用校验与真实 Dataset preflight"),
    Stage("阶段 A3｜第二层验证、原子发布与当前结果", (
        Panel("Reference verification", "recon", ("inventory 记录 relative path / size / SHA-256", "full-content 验证全部引用", "扫描 1,407,184 manifest rows")),
        Panel("Real batch preflight", "recon", ("抽取 128 samples", "NPZ→span/crop→FK→369D→normalize→collate", "检查 motion [128,300,369] / text [128,1,4096]")),
        Panel("CURRENT PRODUCT", "product", ("adopted-legacy-soma30-v1", "motions/text-cache/stats/relative manifests", "inventory + resource-state.json", "mode=verified_legacy_no_reencode", "全部成功后 os.replace 原子发布")),
    )),
)


TRAIN_STAGES = (
    Stage("T0｜读图约定：先认清名词，再沿唯一时间轴向下读", (
        Panel("① 阅读顺序", "source", (
            "整张图只按从上到下推进", "每个计算阶段内部也从上到下读取",
            "依次是输入、处理、输出三张卡", "没有斜向连线，也没有跨阶段回跳",
        )),
        Panel("② Shape 字母", "code", (
            "B：单张 GPU 的 micro-batch 大小", "T：当前 batch 的最长动作帧数",
            "D：每帧动作维数；这里固定为 369", "P：文本向量个数；正式缓存通常为 1",
        )),
        Panel("③ 三种 mask 各管一件事", "boundary", (
            "valid_frames：哪些动作帧不是 padding", "text_pad_mask：哪些文本向量有效",
            "motion_mask：哪些 369D 数值是已知约束", "三者 shape 和用途都不同，不能互换",
        )),
    ), edge="进入训练 step：首先只看 DataLoader 真正提供了什么"),
    Stage("T1｜DataLoader batch：这里只有干净目标、文本和长度信息", (
        Panel("输入｜Dataset 单样本", "source", (
            "clean_motion：已归一化的 369D 动作", "text_features：离线缓存的 LLM2Vec 向量",
            "first_heading_angle：首帧水平朝向", "每条动作仍保留自己的真实帧数",
        )),
        Panel("处理｜Collate 变长样本", "recon", (
            "T 取当前 batch 内最大真实长度", "较短动作只在右侧补零",
            "由 lengths 生成 valid_frames", "文本同理生成 text_pad_mask",
        )),
        Panel("输出｜交给 Trainer", "product", (
            "clean_motion　float32 [B,T,369]", "valid_frames　bool [B,T]；True 是真实帧",
            "lengths　int64 [B]；只供 Trainer 采样约束", "模型内部由 valid_frames 重新得到长度",
            "text_features　float32 [B,P,4096]",
            "text_pad_mask　bool [B,P]", "first_heading_angle　float32 [B]",
            "此时还没有噪声，也没有 motion constraint",
        )),
    ), edge="Trainer 读取 optimizer global_step，决定本次使用哪一段课程"),
    Stage("T2｜课程期与条件采样：约束是在 Trainer 内在线生成的", (
        Panel("输入｜当前 step 与 clean motion", "source", (
            "读取 optimizer global_step", "读取 clean_motion 和每条 lengths",
            "准备独立随机数流", "DataLoader 没有提前保存约束结果",
        )),
        Panel("处理｜论文课程 + 本仓约束 recipe", "recon", (
            "课程期 A（论文 Phase 1）：前 500k steps", "A 期无动作约束；network dropout 为 0.1",
            "课程期 B（论文 Phase 2）：后 500k steps", "B 期 network dropout 为 0；开始抽动作约束",
            "B 期：10% 无约束；25% 两类；65% 一类", "五类：全身、末端、root 稀疏、root 连续、脚接触",
            "全身关键帧只给 root、heading 和 joint positions", "末端关键帧给 root、heading 与选中末端 pos / rot",
            "root 类只给 XZ 和可选 heading；脚接触类只给 4D contact",
            "两个课程期都独立执行 10% text conditioning dropout",
        )),
        Panel("输出｜两套条件", "product", (
            "motion_mask　bool [B,T,369]", "True 表示这个动作数值已知且必须保留",
            "observed_motion　float32 [B,T,369]", "已知位置复制 clean 值；其他位置写零",
            "text dropout 会清零 embedding，并把 text mask 置 False", "当前 use_text_mask=false；去条件依赖 embedding 清零",
            "课程期 A 的 motion_mask 全 False",
        )),
    ), edge="条件已经确定；下一步只负责把 clean target 扰动成 diffusion 输入"),
    Stage("T3｜DDPM forward：由干净目标产生当前噪声版本", (
        Panel("输入｜干净训练目标", "source", (
            "clean_motion 记作 clean x0", "shape 是 [B,T,369]",
            "valid_frames 仍标记真实帧", "clean x0 始终保留，稍后用于监督",
        )),
        Panel("处理｜采样噪声等级", "paper", (
            "每个样本采一个 timestep", "再采同 shape 的高斯噪声",
            "cosine schedule 决定保留多少 clean 信号", "q_sample 混合 clean motion 与噪声",
        )),
        Panel("输出｜noisy motion", "product", (
            "noisy_motion 记作 x_t", "float32 [B,T,369]",
            "timesteps　int64 [B]", "训练目标仍是 clean x0，不是高斯噪声",
        )),
    ), edge="进入 TwostageDenoiser；先把已知约束覆盖回 noisy motion"),
    Stage("T4｜Constraint overwrite：明确哪些值可信，再交给两阶段模型", (
        Panel("输入｜噪声与约束", "product", (
            "noisy_motion　[B,T,369]", "observed_motion　[B,T,369]",
            "motion_mask　[B,T,369]", "mask=True 的位置拥有 clean 观测值",
        )),
        Panel("处理｜逐元素覆盖", "code", (
            "mask=True：使用 observed clean value", "mask=False：继续使用 noisy value",
            "这个操作只替换数值，不删除任何通道", "完整 motion_mask 仍单独传给模型",
        )),
        Panel("输出｜imputed motion", "product", (
            "imputed_motion　float [B,T,369]", "其中已知位置干净，未知位置带噪声",
            "后续 Root 和 Body 都读取同一份 imputed motion", "valid_frames 只负责屏蔽右侧 padding",
        )),
    ), edge="模型第 1 段开始：构造每帧 738D Root 输入和 52 个条件 token"),
    Stage("T5｜模型第 1 段输入组装：完整动作线索 + 完整约束标记", (
        Panel("输入｜每帧两组 369D", "source", (
            "imputed_motion：当前可见动作数值", "motion_mask：逐帧逐通道可信标记",
            "Root 模型段看完整 body 线索", "不是只截取前 5D root 特征",
        )),
        Panel("处理｜构造 motion 与 prefix tokens", "code", (
            "每帧拼接 369D motion 与 369D mask", "得到 738D，再由 Linear 投影到 1024D",
            "文本 pad 到 50 个 slots 并投影到 1024D", "timestep 形成 1 个 token",
            "first heading 形成 1 个 token", "Root 模型段使用自己的一套投影参数",
        )),
        Panel("输出｜Root token sequence", "product", (
            "motion tokens　[B,T,1024]", "prefix tokens　[B,52,1024]",
            "拼接后　[B,52+T,1024]", "valid_frames 控制哪些 motion tokens 可参与 attention",
        )),
    ), edge="Root Transformer 只输出每帧前 5D，但它的输入保留了全部 369D 上下文"),
    Stage("T6｜模型第 1 段：预测 canonical world 中的 global root", (
        Panel("输入｜Root token sequence", "product", (
            "52 个条件 prefix 在前", "T 个 motion tokens 在后",
            "每个 token 宽度都是 1024", "padding 帧不作为 attention key / value",
            "对应输出位置仍会计算；loss 随后用 valid mask 忽略",
        )),
        Panel("处理｜Global-root Transformer", "code", (
            "16 层 Transformer Encoder", "每层 8 个 attention heads",
            "FFN 宽度从 1024 到 2048 再回到 1024", "去掉前 52 个 prefix 输出",
            "对剩余 T 个位置做 Linear 1024→5",
        )),
        Panel("输出｜root prediction", "product", (
            "root_pred　float [B,T,5]", "前 3 维是 normalized smooth root XYZ",
            "后 2 维是 normalized heading cos / sin", "这是模型第 1 段的直接监督输出",
        )),
    ), edge="Body Stage 不直接吃 global5；先经过确定性的 5D→4D 运动学 bridge"),
    Stage("T7｜Global5 → Local4 bridge：把 root 轨迹改写成逐帧运动条件", (
        Panel("输入｜模型第 1 段的 global root", "product", (
            "root_pred　[B,T,5]", "当前仍处在 global-root normalized space",
            "它描述位置、朝向，而不是逐帧速度", "每个样本拥有自己的 canonical world",
        )),
        Panel("处理｜固定公式与固定 stats", "code", (
            "先用 global stats 反归一化", "heading 相邻帧差乘 FPS 得 yaw velocity",
            "root XZ 相邻帧差乘 FPS 得平面速度", "保留 root 的 Y 高度",
            "组成 4D 后再用 local-root stats 归一化", "当前实现的 XZ 速度仍在 canonical world 坐标",
        )),
        Panel("输出｜detached local root", "boundary", (
            "root_local　float [B,T,4]", "依次是 yaw 速度、X 速度、Z 速度、root 高度",
            "训练默认在 no_grad 中计算并 detach", "Body loss 不通过这条 bridge 更新 Root 模型段",
        )),
    ), edge="模型第 2 段开始：local4 与 imputed body364、原始 full mask369 汇合"),
    Stage("T8｜模型第 2 段输入组装：root 条件 + body 数值 + full mask", (
        Panel("输入｜三块逐帧数据", "product", (
            "root_local　[B,T,4]", "imputed_motion 的 body slice　[B,T,364]",
            "原始完整 motion_mask　[B,T,369]", "mask 没有被压缩成 body-only mask",
        )),
        Panel("处理｜组装 Body tokens", "code", (
            "沿最后一维拼成每帧 737D", "Linear 将 737D 投影为 1024D",
            "Body 重新建立 text / time / heading prefix", "prefix 数量仍是 50 + 1 + 1",
            "所有投影层都不与 Root 模型段共享参数",
        )),
        Panel("输出｜Body token sequence", "product", (
            "body motion tokens　[B,T,1024]", "body prefix tokens　[B,52,1024]",
            "拼接后　[B,52+T,1024]", "padding 帧不作为 attention key / value",
            "对应输出位置仍计算；loss 随后忽略",
        )),
    ), edge="独立 Body Transformer 读取预测 root 条件，生成剩余 364D clean body"),
    Stage("T9｜模型第 2 段：预测 body representation 的全部 364D", (
        Panel("输入｜Body token sequence", "product", (
            "52 个 Body 自己的 prefix tokens", "T 个包含 local-root 条件的 motion tokens",
            "宽度统一为 1024", "结构相似不代表与 Root 模型段共享权重",
        )),
        Panel("处理｜Body Transformer", "code", (
            "独立的 16 层 Transformer Encoder", "8 heads；FFN 1024→2048→1024",
            "去掉前 52 个 prefix 输出", "对 T 个 motion hidden 做 Linear 1024→364",
        )),
        Panel("输出｜body prediction", "product", (
            "body_pred　float [B,T,364]", "关节位置 90D；global rotation 180D",
            "关节速度 90D；脚接触 4D", "它不重复输出模型第 1 段的 global root 5D",
        )),
    ), edge="把模型第 1 段的 global5 与模型第 2 段的 body364 合并成完整预测"),
    Stage("T10｜合并与七项监督：padding 计算后由 valid mask 排除", (
        Panel("输入｜两个 stage 的预测", "product", (
            "root_pred　[B,T,5]", "body_pred　[B,T,364]",
            "clean target　[B,T,369]", "valid_frames　[B,T]",
        )),
        Panel("处理｜还原物理量并计算 loss", "recon", (
            "拼接得到 predicted clean motion 369D", "按固定 stats 反归一化到物理域",
            "六项表示 loss 分别检查 root、heading、pose", "还检查 rotation、velocity 和 foot contact",
            "FK 项使用预测 rotation 与 target-derived root position", "loss-side FK 会计算完整 padded T，之后再按 mask 归约",
            "所有 numerator 只累计 valid_frames=True 的帧",
        )),
        Panel("输出｜total loss 与梯度去向", "boundary", (
            "root position / heading 直接训练 Root 模型段", "body 表示项和 FK 项训练 Body 模型段",
            "bridge detach 阻断 body loss 回传 Root 模型段", "各项加权后得到本次 micro-batch loss numerator",
        )),
    ), edge="最后在梯度累积边界统一有效帧 denominator，并执行一次 optimizer update"),
    Stage("T11｜反向传播与更新：micro-batch、DDP、EMA 在这里收尾", (
        Panel("输入｜当前 micro-batch loss", "source", (
            "每个 rank 独立完成 forward 和 loss numerator", "同时保留本 rank 的 valid-frame count",
            "非累计边界暂不进行 DDP 梯度同步", "课程阶段依据 optimizer step，而不是 batch index",
        )),
        Panel("处理｜累计与多卡同步", "recon", (
            "非边界 micro-step 使用 DDP no_sync", "累计边界 all-reduce 全局有效帧数",
            "用同一个全局 denominator 缩放梯度", "随后执行 gradient clip；阈值为 1.0",
        )),
        Panel("输出｜一次 optimizer step", "product", (
            "Adam-atan2 更新 Root 与 Body 参数", "EMA 每 10 个 optimizer steps 更新",
            "checkpoint 保存模型、优化器、EMA、step 与 RNG", "正式 overlay：两卡×每卡128×累计8次=2048",
        )),
    )),
)


def render_readable_timeline(
    output: Path,
    *,
    title: str,
    subtitle: str,
    stages: tuple[Stage, ...],
    conclusion: tuple[str, str],
) -> None:
    """Render a 900px-readable, strictly top-to-bottom technical timeline."""

    width = 1800
    margin = 70
    content_w = width - 2 * margin
    card_x = margin + 50
    card_w = content_w - 100
    card_gap = 62
    header_h = 570
    edge_h = 132
    line_step = 51
    stage_heights: list[int] = []
    panel_heights: list[list[int]] = []
    for stage in stages:
        heights = [170 + len(panel.lines) * line_step for panel in stage.panels]
        panel_heights.append(heights)
        stage_heights.append(130 + sum(heights) + card_gap * (len(heights) - 1) + (65 if stage.note else 30))
    height = 45 + header_h + sum(stage_heights) + edge_h * (len(stages) - 1) + 180

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f'<title>{escape(title)}</title>',
        f'<desc>{escape(subtitle)}。所有阶段从上到下排列，阶段内部从输入到处理再到输出。</desc>',
        f'<rect width="{width}" height="{height}" fill="#F8FAFC"/>',
        '<defs><marker id="arrow" markerWidth="20" markerHeight="20" viewBox="0 0 20 20" refX="17" refY="10" orient="auto" markerUnits="userSpaceOnUse"><path d="M1,1 L18,10 L1,19 z" fill="#475569"/></marker></defs>',
        _text(width / 2, 92, title, 48, 900, "#0F172A"),
        _text(width / 2, 148, subtitle, 26, 520, "#334155"),
    ]

    rules = (
        ("只向下读", "外层阶段严格按真实执行顺序排列"),
        ("卡片也向下", "每个阶段固定为输入、处理、输出"),
        ("先解释再使用", "shape、mask 和梯度边界都会先定义"),
    )
    for index, (rule_title, rule_text) in enumerate(rules):
        y = 195 + index * 92
        out.append(f'<rect x="{margin + 55}" y="{y}" width="{content_w - 110}" height="72" rx="22" fill="#FFFFFF" stroke="#94A3B8" stroke-width="3"/>')
        out.append(_text(margin + 280, y + 46, rule_title, 26, 850, "#0F172A"))
        out.append(_text(width / 2 + 145, y + 46, rule_text, 25, 520, "#475569"))

    legend = (("输入", "source"), ("代码", "code"), ("论文", "paper"),
              ("本仓", "recon"), ("产物", "product"), ("边界", "boundary"))
    legend_w = 245
    legend_gap = 28
    legend_x = (width - (len(legend) * legend_w + (len(legend) - 1) * legend_gap)) / 2
    for index, (label, kind) in enumerate(legend):
        fill, stroke, _ = COLORS[kind]
        x = legend_x + index * (legend_w + legend_gap)
        out.append(f'<rect x="{x:.0f}" y="493" width="{legend_w}" height="50" rx="25" fill="{fill}" stroke="{stroke}" stroke-width="3"/>')
        out.append(_text(x + legend_w / 2, 527, label, 23, 850, stroke))

    y = 45 + header_h
    for stage_index, (stage, stage_h, heights) in enumerate(zip(stages, stage_heights, panel_heights, strict=True)):
        out.append(f'<rect x="{margin}" y="{y}" width="{content_w}" height="{stage_h}" rx="30" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="4"/>')
        badge_fill = "#173D77" if stage_index < 5 else "#31583A"
        out.append(f'<rect x="{margin + 30}" y="{y + 25}" width="160" height="54" rx="27" fill="{badge_fill}"/>')
        out.append(_text(margin + 110, y + 62, f"STEP {stage_index:02d}", 22, 900, "#FFFFFF"))
        out.append(_text(width / 2 + 60, y + 61, stage.title, 31, 880, "#0F172A"))

        panel_y = y + 105
        for panel_index, (panel, panel_h) in enumerate(zip(stage.panels, heights, strict=True)):
            fill, stroke, evidence = COLORS[panel.kind]
            if panel.kind == "source":
                evidence = "INPUT / CONTEXT"
            dash = ' stroke-dasharray="15 11"' if panel.kind == "boundary" else ""
            out.append(f'<rect x="{card_x}" y="{panel_y}" width="{card_w}" height="{panel_h}" rx="24" fill="{fill}" stroke="{stroke}" stroke-width="4"{dash}/>')
            out.append(_text(width / 2, panel_y + 39, evidence, 21, 850, stroke))
            out.append(_text(width / 2, panel_y + 86, panel.title, 32, 850, "#111827"))
            text_y = panel_y + 137
            for line in panel.lines:
                out.append(_text(width / 2, text_y, line, 28, 470, "#172033"))
                text_y += line_step
            panel_y += panel_h
            if panel_index < len(stage.panels) - 1:
                arrow_end = panel_y + card_gap - 16
                out.append(f'<path d="M{width/2:.0f} {panel_y + 8} L{width/2:.0f} {arrow_end}" stroke="#475569" stroke-width="7" fill="none"/>')
                out.append(f'<polygon points="{width/2-14:.0f},{arrow_end-15:.0f} {width/2+14:.0f},{arrow_end-15:.0f} {width/2:.0f},{arrow_end+7:.0f}" fill="#475569"/>')
                panel_y += card_gap

        if stage.note:
            out.append(_text(width / 2, y + stage_h - 24, stage.note, 25, 700, "#475569"))

        y += stage_h
        if stage_index < len(stages) - 1:
            out.append(f'<path d="M{width/2:.0f} {y + 8} L{width/2:.0f} {y + 105}" stroke="#475569" stroke-width="8" fill="none"/>')
            if stage.edge:
                out.append(f'<rect x="{margin + 120}" y="{y + 31}" width="{content_w - 240}" height="50" rx="25" fill="#F8FAFC"/>')
                out.append(_text(width / 2, y + 66, stage.edge, 24, 720, "#475569"))
            out.append(f'<polygon points="{width/2-15:.0f},{y+90} {width/2+15:.0f},{y+90} {width/2:.0f},{y+116}" fill="#475569"/>')
            y += edge_h

    out.append(f'<rect x="{margin}" y="{y + 35}" width="{content_w}" height="105" rx="24" fill="#E8F7EE" stroke="#24815A" stroke-width="4"/>')
    out.append(_text(width / 2, y + 75, conclusion[0], 27, 850, "#14532D"))
    out.append(_text(width / 2, y + 115, conclusion[1], 25, 620, "#14532D"))
    out.append("</svg>")
    output.write_text("\n".join(out) + "\n", encoding="utf-8")


MODEL_STAGES = (
    Stage("阶段 M0｜TwostageDenoiser.forward 的完整输入合同", (
        Panel("当前动作与观测", "product", (
            "x：noisy motion float [B,T,369]", "motion_mask bool [B,T,369]", "observed_motion float [B,T,369]",
            "369 = global-root 5 + body 364；T≤300",
        )),
        Panel("序列与扩散条件", "product", (
            "x_pad_mask bool [B,T]；True=有效帧", "timesteps int [B]；范围 0…999",
            "first_heading_angle float [B]", "lengths = x_pad_mask.sum(-1) → [B]",
        )),
        Panel("文本条件", "product", (
            "text_feat float [B,1,4096]", "text_feat_pad_mask bool [B,1]",
            "1 个离线 LLM2Vec 句向量；不是 token IDs", "同一输入分别进入 root/body 各自的条件投影层",
        )),
    ), edge="constraint overwrite：先把 mask=True 的位置替换成 clean observed value"),
    Stage("阶段 M1｜进入模型后的 imputation 与 Root-stage 逐帧输入", (
        Panel("Clean observation overwrite", "code", (
            "motion_mask → float mask features", "x̃ = where(mask, observed_motion, x)",
            "x̃ float [B,T,369]", "未提供 mask/observed 时，两者自动视为全零",
        )),
        Panel("Root motion input concat", "code", (
            "x̃ [B,T,369]", "+ full motion mask [B,T,369]", "沿最后一维 concatenate",
            "root input = [B,T,738]",
        )),
        Panel("为什么不是只送 root 5D", "boundary", (
            "Root Transformer 看完整 noisy/imputed 369D 动作", "也看完整 369D observation mask",
            "body pose / contact 可为 root 轨迹提供上下文", "只把最终输出头限制成 5D global root",
        )),
    ), edge="Root block 分成 motion-token 分支与 3 个 prefix-condition 分支，全部投影到 latent=1024"),
    Stage("阶段 M2｜Root Transformer 的四条输入分支及精确维度", (
        Panel("Motion token projection", "code", (
            "root input [B,T,738]", "Linear(738→1024) independently per frame",
            "motion tokens [B,T,1024]", "有效性仍由 x_pad_mask [B,T] 控制",
        )),
        Panel("Text prefix：50 tokens", "code", (
            "[B,1,4096] 先 zero-pad 到 [B,50,4096]", "Linear(4096→1024)",
            "text tokens [B,50,1024]", "use_text_mask=false → 50 个 slot 的 attention mask 全 True",
        )),
        Panel("Time + first-heading prefix", "code", (
            "timestep [B] → sinusoidal PE lookup [B,1,1024]", "→ Linear1024→1024 + SiLU + Linear1024→1024",
            "angle [B] → [cos,sin] [B,2] → Linear2→1024", "time 1 token + heading 1 token = [B,2,1024]",
        )),
    ), edge="按 text 50 → time 1 → heading 1 → motion T 的顺序拼成一条 token sequence"),
    Stage("阶段 M3｜Root TransformerEncoderBlock 内部：长度 52+T，宽度始终 1024", (
        Panel("Sequence + attention mask", "code", (
            "prefix [B,52,1024] + motion [B,T,1024]", "xseq = [B,52+T,1024]；T≤300 → 长度≤352",
            "valid mask = [text50=True,time=True,heading=True,x_pad_mask]", "PyTorch src_key_padding_mask 对上述 bool 取反",
        )),
        Panel("16 × TransformerEncoderLayer", "code", (
            "8-head self-attention；head dimension=1024/8=128", "FFN：1024→2048→1024；activation=GELU",
            "batch_first=true；norm_first=false，即 post-norm", "network dropout：课程期 A 为 .1；课程期 B 为 0",
        )),
        Panel("只保留 motion positions", "code", (
            "encoder output [B,52+T,1024]", "丢弃前 52 个 prefix outputs",
            "motion hidden = output[:,52:] → [B,T,1024]", "Linear(1024→5) → root_pred [B,T,5]",
        )),
    ), edge="5D 仍是 normalized global-root；先回物理量生成局部速度，再归一化为 body condition"),
    Stage("阶段 M4｜Root 输出与 5D→4D bridge：通道减少不是 Linear，而是确定性运动学变换", (
        Panel("Stage-1 global output", "product", (
            "root_pred [B,T,5]", "通道：[root x,y,z, heading cosθ,sinθ]",
            "位于 global-root mean/std 定义的 normalized space", "最终 369D 输出的前 5 维直接来自这里",
        )),
        Panel("global_root_to_local_root", "code", (
            "global stats unnormalize：5D → 物理量", "atan2(sin,cos) + wrapped temporal difference × FPS",
            "canonical-world ΔXZ×FPS；不再旋转进 heading frame", "组合 [yaw velocity, canonical vx, canonical vz, root y]",
            "local-root stats normalize → root_local [B,T,4]",
        )),
        Panel("Training gradient boundary", "boundary", (
            "detach_root_for_body=true", "training：bridge 位于 no_grad，并对 local4 再 detach",
            "body loss 不通过 local4 回传 Root Transformer", "eval/guidance：不强制 no_grad；可保留跨 bridge 梯度",
        )),
    ), edge="从已经 imputed 的 x̃ 只取 body364，再拼 root_local4 与完整 mask369"),
    Stage("阶段 M5｜Body-stage 输入：4 + 364 + 369 = 737 channels / frame", (
        Panel("Body motion branch", "code", (
            "body_x = x̃[..., body_slice]", "x̃ [B,T,369] → body_x [B,T,364]",
            "concat[root_local4, body_x364]", "local-conditioned motion [B,T,368]",
        )),
        Panel("Append original full mask", "code", (
            "local-conditioned motion [B,T,368]", "+ full mask features [B,T,369]",
            "body input [B,T,737]", "mask 仍覆盖原 369D 语义，而不是压缩成 368D mask",
        )),
        Panel("Body block 不共享参数", "boundary", (
            "独立 input/output Linear", "独立 text Linear、time MLP、heading Linear",
            "独立 16-layer TransformerEncoder", "与 Root block 架构相同，但所有可训练参数分离",
        )),
    ), edge="Body block 重新构造自己的 52 个 prefix tokens，并将 737D motion 投影到 1024D"),
    Stage("阶段 M6｜Body TransformerEncoderBlock 的维度流", (
        Panel("Body token projections", "code", (
            "motion：Linear(737→1024) → [B,T,1024]", "text：pad 1→50；Linear(4096→1024)",
            "time：PE lookup + 1024→1024→1024 MLP", "heading：[cos,sin] 2→1024",
        )),
        Panel("Independent encoder", "code", (
            "concat → [B,52+T,1024]", "positional encoding + Phase-dependent dropout",
            "16 layers / 8 heads / head dim128 / FFN2048", "slice prefix → motion hidden [B,T,1024]",
        )),
        Panel("Stage-2 body output", "product", (
            "Linear(1024→364)", "predicted_body [B,T,364]",
            "顺序严格对应 joint pos90 / rotation180", "velocity90 / heel-toe contact4",
        )),
    ), edge="沿 feature dimension 拼接 Stage-1 global5 与 Stage-2 body364"),
    Stage("阶段 M7｜TwostageDenoiser 最终输出", (
        Panel("Concatenation", "code", (
            "root_pred [B,T,5]", "+ predicted_body [B,T,364]", "output = [B,T,369]",
        )),
        Panel("输出语义", "product", (
            "前 5D：global root XYZ + heading cos/sin", "后 364D：body representation",
            "shape 与输入 noisy motion x 完全一致", "trainer 将其解释为 clean x₀ prediction",
        )),
        Panel("模型本身不包含", "boundary", (
            "不包含 LLM2Vec 8B 编码器；只接收缓存 embedding", "不包含 DDPM q_sample 或七项 loss",
            "不包含 optimizer / EMA / DDP", "这些位于模型外层 trainer，已在训练阶段 SVG 展示",
        )),
    )),
)


# A second, cache-busting presentation asset whose cards use an explicit
# “tensor + shape —— semantic description” contract.  Keep the compact
# MODEL_STAGES above for existing links; new documentation points here.
MODEL_DETAILED_STAGES = (
    Stage("D0｜模型入口：先把每个张量的物理含义和 mask 语义说清楚", (
        Panel("动作、观测值与 feature-level constraint mask", "product", (
            "x  float32 [B,T,369] —— 当前 diffusion step 的 noisy motion x_t；它不是 clean ground truth",
            "B —— 当前 GPU/rank 的 micro-batch 样本数；不同 b 是互相独立的 motion clips",
            "T —— batch padding 后的帧数；每帧 1/30 秒；训练最大 T=300，对应 10 秒",
            "369 —— 每帧动作宽度：global-root 5D + body 364D",
            "observed_motion  float32 [B,T,369] —— constraint 给出的 clean x_0 数值容器",
            "motion_mask  bool [B,T,369] —— True 表示同位置 observed_motion 值可信并必须覆盖 noisy x",
            "motion_mask=False 时 observed_motion 对应元素不读取；通常是 0，但语义完全由 mask 决定",
        )),
        Panel("frame-level padding、diffusion step 与首帧朝向", "product", (
            "x_pad_mask  bool [B,T] —— True=真实动作帧；False=为对齐 batch 长度而补的 padding frame",
            "区别 —— x_pad_mask 屏蔽整帧 attention；motion_mask 标记某帧某个 feature 是否被约束",
            "timesteps  int64 [B] —— 每个样本的 diffusion step，范围 0…999，用来表达当前噪声强度",
            "同一个样本的全部 T 帧共享一个 timestep；不同 batch 样本可以抽到不同 timestep",
            "first_heading_angle  float32 [B] —— canonicalization 选定的首帧水平朝向，单位 radians",
            "lengths  int64 [B] —— forward 内由 x_pad_mask.sum(-1) 得到；不是额外的数据集字段",
            "lengths 用于速度差分的序列尾部处理，防止把 padding frame 当成下一帧",
        )),
        Panel("离线文本条件", "product", (
            "text_feat  float32 [B,1,4096] —— 每条 caption 的一个离线 LLM2Vec sentence embedding",
            "第二维 1 —— 一个句向量槽位，不是自然语言只有一个 token；文字已被整体编码",
            "最后一维 4096 —— LLM2Vec embedding width；不是 motion feature width",
            "text_feat_pad_mask  bool [B,1] —— 输入句向量槽位是否有效",
            "动作训练不加载 8B LLM；它只读取数据预处理阶段缓存的 embedding",
            "同一 text_feat 会进入 Root 和 Body 各自的 Linear(4096→1024)",
            "输入数值相同，但两套 projection parameters 不共享，所以产生的 prefix hidden 不同",
        )),
    ), edge="逐元素 constraint overwrite：只替换 motion_mask=True 的 feature；shape [B,T,369] 保持不变"),

    Stage("D1｜Constraint overwrite 与 738D Root 输入：公式中的每一项如何工作", (
        Panel("torch.where：在 feature 粒度混合 clean observation 与 noisy motion", "code", (
            "输入 x / observed_motion / motion_mask —— 三者 shape 都是 [B,T,369]，没有 broadcasting",
            "motion_mask_bool = motion_mask.bool() —— 强制把条件解释为开关，而不是连续权重",
            "当 mask[b,t,d]=True：x_tilde[b,t,d] = observed_motion[b,t,d]，即 clean x_0 值",
            "当 mask[b,t,d]=False：x_tilde[b,t,d] = x[b,t,d]，即保留 noisy x_t 值",
            "输出 x_tilde  float32 [B,T,369] —— clean constrained features 与 noisy unknown features 共存",
            "mask_float = motion_mask_bool.to(x.dtype) —— bool mask 转成 0/1 float [B,T,369]",
            "若 mask 或 observed 任一未提供，公开 forward 把两者都置零；于是 x_tilde=x，表示无 constraint",
        )),
        Panel("为什么拼接后是 738D，而不是把 mask 乘进 motion 就结束", "code", (
            "左半 x_tilde [B,T,369] —— 网络看到的实际动作数值",
            "右半 mask_float [B,T,369] —— 网络知道左半每个值是 clean observation 还是 noisy estimate",
            "torch.cat([x_tilde,mask_float],dim=-1) —— 只沿 feature 轴拼接",
            "B 不变，T 不变，feature width 从 369 变成 369+369=738",
            "root_input  float32 [B,T,738] —— Root Transformer 的逐帧输入合同",
            "此处没有 temporal mixing、loss、normalization 或 averaging；只是显式保留值与可信度",
            "如果不输入 mask，数值相同的 clean observation 与随机 noise 对网络来说无法区分来源",
        )),
        Panel("Root 为什么读取完整 body 上下文，却只输出 5D", "code", (
            "root_input 左半仍含 joint position、rotation、velocity 和 foot-contact 等完整 body features",
            "例如脚接触和腿部速度能帮助判断 pelvis 应该静止、平移还是转弯",
            "完整 369D mask 还能说明这些 body 线索是否来自用户/constraint 的 clean observation",
            "网络没有在输入端裁成 root5；它让 self-attention 使用全部动作上下文",
            "只有 Root block 最后的 output Linear 把 hidden width 1024 映射为 global-root 5D",
            "因此 738D 是上下文输入宽度，5D 是 Stage-1 clean-root 预测目标宽度",
        )),
    ), edge="Root block 把 motion、text、timestep、heading 四条分支分别投影成 width=1024 的 tokens"),

    Stage("D2｜四条 token 分支：Linear 改了什么，哪些维度没有改变", (
        Panel("Motion tokens：738 个 frame features → 1024 hidden channels", "code", (
            "root_input [B,T,738] —— 每个 t 含 x_tilde369 与对应 mask369",
            "input_linear = Linear(738→1024) —— 独立作用于每个 frame 的最后一维",
            "操作只在单帧内混合 channels；在进入 Transformer 前，frame t 还看不到其他时间帧",
            "输出 motion_tokens [B,T,1024] —— T 个动作 token，每个 token hidden width=1024",
            "B 和 T 不改变；x_pad_mask 也不进入 Linear，而是在 attention 阶段屏蔽 padding",
        )),
        Panel("Text tokens：1 个 sentence embedding 为什么被补成 50 个 slots", "code", (
            "输入 text_feat [B,1,4096] —— 一句话的整体语义向量",
            "pad_x_and_mask_to_fixed_size 在 token 轴补 49 个 zero slots → [B,50,4096]",
            "这不是重新 tokenize 文本；新增位置最初只是为了 checkpoint-compatible 固定长度",
            "embed_text = Linear(4096→1024) 分别投影 50 个 slots → text_tokens [B,50,1024]",
            "Linear 有 bias，所以 zero slot 投影后可成为相同的 learned bias vector，而不保证仍是零",
            "use_text_mask=false 会把 50 个 slot mask 全设为 True；它们全部参与 self-attention",
        )),
        Panel("Time token 与 heading token", "code", (
            "timesteps [B] —— 用整数 step 索引固定 sinusoidal positional table → [B,1,1024]",
            "time MLP —— Linear1024→1024、SiLU、Linear1024→1024；输出 1 个 time token",
            "first_heading_angle [B] —— 先计算 cos/sin → [B,2]，避免角度在 ±π 处数值跳变",
            "heading Linear(2→1024) 并增加 token 轴 → [B,1,1024]",
            "time_mask 与 heading_mask 固定 True，因为每个样本必须提供这两个条件",
            "条件 token 总数 —— text50 + time1 + heading1 = prefix52，shape [B,52,1024]",
        )),
    ), edge="在 token 轴按 text50、time1、heading1、motionT 拼接；得到 [B,52+T,1024]"),

    Stage("D3｜TransformerEncoderBlock：条件怎样真正影响每一帧动作", (
        Panel("Sequence、position 与 attention padding mask", "code", (
            "prefix [B,52,1024] —— 携带 text/time/heading 条件，不对应任何动作帧",
            "motion_tokens [B,T,1024] —— 后 T 个位置与原 motion frames 一一对应",
            "cat(dim=1) → xseq [B,52+T,1024]；T≤300，所以最大 sequence length=352",
            "固定 sinusoidal positional encoding 加到每个位置，让 attention 区分 prefix 顺序和时间顺序",
            "valid_mask = [text50 True,time True,heading True,x_pad_mask]",
            "PyTorch src_key_padding_mask 语义相反，因此代码对 valid_mask 取逻辑非",
            "motion_mask 不在这里屏蔽 attention；它已经作为 float 0/1 channels 拼进 motion token",
        )),
        Panel("16 层 self-attention 与 FFN 在做什么", "code", (
            "每层 8-head self-attention；hidden1024/8=128，因此每个 attention head width=128",
            "每个 motion frame 可以读取 52 个条件 tokens 和全部其他有效 motion frames",
            "所以文字语义、噪声强度、首帧朝向和长时动作上下文都能影响当前帧 hidden",
            "prefix tokens 也被更新，但它们的输出稍后丢弃；价值在于向 motion positions 传播条件",
            "FFN 逐 token 执行 1024→2048→1024，activation=GELU",
            "norm_first=false 表示 post-norm；batch_first=true 保持 [B,sequence,hidden]",
            "课程期 A attention/FFN/PE dropout=.1；课程期 B 将这些 dropout 改为 0",
        )),
        Panel("为什么 slice 前 52 个输出，再做 1024→5", "code", (
            "Transformer 输出 [B,52+T,1024] —— sequence length 和 hidden width 都没有改变",
            "前 52 个位置只是条件载体，没有对应 ground-truth motion frame，不直接生成动作 feature",
            "output[:,52:] → motion_hidden [B,T,1024]，只保留和 T 个动作帧对齐的位置",
            "Root output_linear = Linear(1024→5) 独立映射每个 motion hidden",
            "root_pred [B,T,5] —— normalized clean x_0 root 预测，不是 noisy input root 的复制",
            "5D 顺序 —— smooth-root X/Y/Z 3D + heading cos/sin 2D",
        )),
    ), edge="root_pred5 一路直接进入最终输出；另一路通过确定性 5D→4D bridge 形成 Body condition"),

    Stage("D4｜5D global root → 4D local-root condition：不是 Linear，而是物理时间差分", (
        Panel("root_pred 五个通道及其坐标语义", "product", (
            "root_pred [B,T,5] —— 仍位于 global-root mean/std 定义的 normalized space",
            "通道 0/1/2 —— 每条样本 canonical-world 中的 smooth-root X/Y/Z",
            "通道 3/4 —— heading cosθ/sinθ；用二维单位圆表示避免 ±π discontinuity",
            "不同 batch 样本各有自己的 canonical origin/heading，并不共享真实物理场景世界坐标",
            "这 5D 保留为最终 output 前五维；bridge 只是额外派生 Body 所需的条件",
        )),
        Panel("global_root_to_local_root 的六个实际步骤", "code", (
            "① global stats unnormalize —— 把 5D 从 z-score 还原为位置与 heading 数值",
            "② atan2(sin,cos) → heading angle；wrapped 相邻帧角差 ×30 FPS → yaw velocity",
            "③ canonical-world X/Z 相邻帧位置差 ×30 FPS → vx、vz",
            "公开代码没有把 vx/vz 再旋入 heading frame；local_root 是表示名，不代表 heading-local 坐标",
            "④ 每帧直接保留 root Y，表示绝对离地高度；Y 不做时间差分",
            "⑤ 拼成 physical local4 = [yaw velocity,vx,vz,root Y]，shape [B,T,4]",
            "⑥ 用固定全训练集 local-root mean/std normalize → root_local [B,T,4]",
            "为保持 T 不变，最后有效帧速度复制倒数第二帧；lengths 避免差分读到 padding",
        )),
        Panel("detach 的具体后果", "boundary", (
            "detach_root_for_body=true —— 当前配置与公开 training-mode forward 一致",
            "训练时 bridge 位于 torch.no_grad()，随后 root_local 再执行 .detach()",
            "Body block 能读取 local4 数值，但 autograd 不保留 local4→root_pred 的计算路径",
            "Body representation/FK loss 因此不会穿过 bridge 更新 Root Transformer",
            "Root Transformer 仍由最终 output 前 5D 的 root-position 与 heading loss 直接训练",
            "eval/guidance 不进入 training detach 分支，可保留跨 bridge gradient 供 guidance 使用",
        )),
    ), edge="Body input 使用 detached root_local4 + overwrite 后的 body364 + 原始 full mask369"),

    Stage("D5｜Body 输入与第二个 Transformer：4+364+369 为什么正好是 737", (
        Panel("Body input 的三个来源", "code", (
            "body_x = x_tilde[...,body_slice] —— 从 overwrite 后的动作取 indices 5…368",
            "body_x [B,T,364] —— constraint 位置为 clean observation；其他位置仍是 noisy x_t",
            "root_local [B,T,4] —— 来自预测 root5 的时间差分，并已按 local stats normalize",
            "先拼 root_local4 + body_x364 → local-conditioned motion [B,T,368]",
            "再拼原始 mask_float [B,T,369]；mask 仍按 global5+body364 的原表示排列",
            "body_input = cat([local4,body364,mask369],dim=-1) → [B,T,737]",
            "这一设计用预测 root 的速度条件替代 noisy global-root5，同时保留全部 constraint provenance",
        )),
        Panel("Body block 与 Root block 的相同点和独立参数", "code", (
            "Body input_linear = Linear(737→1024)；Root 对应层为 Linear(738→1024)",
            "Body 自己重新计算 text50、time1、heading1；不会复用 Root prefix hidden",
            "两者超参数相同：latent1024、16 layers、8 heads、FFN2048、prefix52",
            "但 embed_text、time MLP、heading Linear、Transformer layers 和 output Linear 全部不共享",
            "Body token sequence 同样为 [B,52+T,1024]，并使用同一个 x_pad_mask 语义",
            "经过 self-attention 后丢弃 prefix outputs → body_hidden [B,T,1024]",
        )),
        Panel("Body 364D 输出的逐块语义", "product", (
            "Body output_linear = Linear(1024→364)；输出 predicted_body [B,T,364]",
            "body indices 0…89 —— 30 joints × XYZ position = 90D",
            "body indices 90…269 —— 30 joints × global rotation 6D = 180D",
            "body indices 270…359 —— 30 joints × XYZ velocity = 90D",
            "body indices 360…363 —— left/right heel/toe contact = 4D",
            "predicted_body 是 normalized clean x_0 body slice，而不是预测 diffusion noise ε",
        )),
    ), edge="最终不再经过第三个网络；只沿 feature 轴拼 root clean5 与 body clean364"),

    Stage("D6｜最终输出：每个维度是什么，以及哪些工作在模型外", (
        Panel("Feature concat 与最终 shape", "code", (
            "root_pred [B,T,5] —— Stage-1 clean global-root prediction",
            "predicted_body [B,T,364] —— Stage-2 clean body prediction",
            "output = torch.cat([root_pred,predicted_body],dim=-1)",
            "只沿 feature 轴拼接：5+364=369；B、T 都不改变",
            "最终 output float32/BF16 [B,T,369]，shape 与输入 noisy x 完全一致",
            "这种 shape 对齐让 trainer 能逐帧逐 feature 和 clean_motion target 比较",
        )),
        Panel("最终 369D feature map", "product", (
            "indices 0…2 —— smooth global-root XYZ；indices 3…4 —— heading cos/sin",
            "indices 5…94 —— joint positions90；indices 95…274 —— global rotations180",
            "indices 275…364 —— joint velocities90；indices 365…368 —— heel/toe contacts4",
            "output 是 normalized clean-motion x_hat_0；不是已经 inverse/FK 后的 BVH/NPZ",
            "padding frames 虽占有 tensor entries，但必须由 valid_frames/x_pad_mask 从 loss 中排除",
        )),
        Panel("TwostageDenoiser 外部职责", "boundary", (
            "8B LLM2Vec 不在模型内运行 —— text_feat 已由离线数据管线缓存",
            "DDPM q_sample 不在模型内 —— trainer 先产生 noisy x 和 timestep 再调用 forward",
            "六项 representation loss 与 FK loss 不在模型内 —— forward 只返回 [B,T,369] prediction",
            "backward、gradient accumulation、DDP all-reduce、optimizer、EMA、checkpoint 均属于 trainer",
            "这些步骤应在训练流程图解释，不能误画成 TwostageDenoiser 的神经网络层",
        )),
    )),
)


def render_model_io_architecture(output: Path) -> None:
    """Render a landscape block diagram with explicit tensor inputs/outputs."""

    width, height = 6500, 3650
    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f'<rect width="{width}" height="{height}" fill="#F8FAFC"/>',
        '<defs><marker id="arrow" markerWidth="18" markerHeight="18" refX="13" refY="7" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L14,7 L0,14 z" fill="#475569"/></marker><marker id="redarrow" markerWidth="18" markerHeight="18" refX="13" refY="7" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L14,7 L0,14 z" fill="#C24132"/></marker></defs>',
        _text(width / 2, 105, "Kimodo TwostageDenoiser｜模型输入、内部结构与输出维度", 57, 900, "#0F172A"),
        _text(width / 2, 165, "沿箭头读取：369D noisy motion → Root Transformer → global5 → bridge local4 → Body Transformer → body364 → clean motion369", 28, 500, "#334155"),
    ]

    def box(x: int, y: int, w: int, h: int, title: str, lines: tuple[str, ...], kind: str,
            title_size: int = 28, line_size: int = 23, dashed: bool = False) -> None:
        fill, stroke, evidence = COLORS[kind]
        dash = ' stroke-dasharray="15 11"' if dashed else ""
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="22" fill="{fill}" stroke="{stroke}" stroke-width="4"{dash}/>')
        out.append(_text(x + w / 2, y + 31, evidence, 17, 850, stroke))
        out.append(_text(x + w / 2, y + 72, title, title_size, 850, "#111827"))
        line_y = y + 111
        for line in lines:
            out.append(_text(x + w / 2, line_y, line, line_size, 450, "#172033"))
            line_y += 37

    def arrow(points: tuple[tuple[int, int], ...], label: str = "", label_xy: tuple[int, int] | None = None,
              color: str = "#475569", dashed: bool = False) -> None:
        d = "M" + " L".join(f"{x} {y}" for x, y in points)
        dash = ' stroke-dasharray="14 11"' if dashed else ""
        marker = "redarrow" if color == "#C24132" else "arrow"
        out.append(f'<path d="{d}" stroke="{color}" stroke-width="7" fill="none" marker-end="url(#{marker})"{dash}/>')
        if label and label_xy:
            label_w = max(210, len(label) * 15)
            lx, ly = label_xy
            out.append(f'<rect x="{lx-label_w/2:.0f}" y="{ly-29}" width="{label_w}" height="40" rx="18" fill="#F8FAFC"/>')
            out.append(_text(lx, ly, label, 20, 750, color))

    # Legend
    legends = (("INPUT", "source"), ("CODE / CONFIG", "code"), ("TENSOR", "product"), ("DETACH / BOUNDARY", "boundary"))
    lx = 1450
    for label, kind in legends:
        fill, stroke, _ = COLORS[kind]
        out.append(f'<rect x="{lx}" y="205" width="760" height="54" rx="27" fill="{fill}" stroke="{stroke}" stroke-width="3"/>')
        out.append(_text(lx + 380, 241, label, 21, 850, stroke))
        lx += 800

    # Raw model inputs: left and top, visually separate from the network.
    out.append('<rect x="75" y="340" width="1120" height="2700" rx="30" fill="#FFFFFF" stroke="#94A3B8" stroke-width="4"/>')
    out.append(_text(635, 395, "MODEL INPUTS", 34, 900, "#0F172A"))
    box(125, 445, 1020, 210, "Noisy motion x", ("float [B,T,369]", "global-root5 + body364；T≤300"), "source")
    box(125, 700, 1020, 210, "Observed clean values", ("observed_motion float [B,T,369]", "只在 motion_mask=True 的通道生效"), "source")
    box(125, 955, 1020, 210, "Observation mask", ("motion_mask bool [B,T,369]", "随后 cast 为 float mask features"), "source")
    box(125, 1270, 1020, 210, "LLM2Vec sentence", ("text_feat float [B,1,4096]", "text_feat_pad_mask bool [B,1]"), "source")
    box(125, 1525, 1020, 190, "Diffusion timestep", ("timesteps int [B]；0…999",), "source")
    box(125, 1760, 1020, 190, "First heading", ("first_heading_angle float [B]",), "source")
    box(125, 1995, 1020, 210, "Valid-frame mask", ("x_pad_mask bool [B,T]", "True=真实帧；False=padding"), "source")
    box(125, 2250, 1020, 250, "模型外输入说明", ("x 已由 DDPM q_sample 产生", "text 已由离线 8B LLM2Vec 产生", "loss / optimizer 不在本模型内部"), "boundary", dashed=True)

    # Shared imputation, then the two main paths.
    box(1320, 475, 900, 300, "Constraint overwrite", (
        "x̃ = where(mask, observed, x)", "x̃ [B,T,369]", "mask_float [B,T,369]",
    ), "code")
    arrow(((1145, 550), (1320, 550)))
    arrow(((1145, 805), (1250, 805), (1250, 625), (1320, 625)))
    arrow(((1145, 1060), (1280, 1060), (1280, 700), (1320, 700)))

    # Shared raw conditions bus and two independent prefix encoders.
    out.append('<rect x="1320" y="850" width="5080" height="190" rx="24" fill="#EAF2FF" stroke="#3973C6" stroke-width="4"/>')
    out.append(_text(3860, 905, "SHARED RAW CONDITIONS（数值相同，但 Root / Body 使用各自独立的投影参数）", 29, 850, "#173D77"))
    out.append(_text(3860, 958, "text [B,1,4096]　|　timestep [B]　|　first heading [B]　|　x_pad_mask [B,T]", 25, 500, "#173D77"))
    arrow(((1145, 1375), (1230, 1375), (1230, 900), (1320, 900)))
    arrow(((1145, 1620), (1210, 1620), (1210, 930), (1320, 930)))
    arrow(((1145, 1855), (1190, 1855), (1190, 960), (1320, 960)))
    arrow(((1145, 2100), (1170, 2100), (1170, 990), (1320, 990)))

    # Root path band.
    out.append('<rect x="1260" y="1080" width="5140" height="880" rx="30" fill="#FFFFFF" stroke="#5C8465" stroke-width="5"/>')
    out.append(_text(3830, 1135, "ROOT PATH｜Global-root Transformer", 36, 900, "#31583A"))
    box(1320, 1200, 700, 250, "Concat motion + mask", ("x̃ 369 + mask 369", "[B,T,738]"), "product")
    box(2130, 1200, 700, 250, "Motion projection", ("Linear 738→1024", "[B,T,1024]"), "code")
    box(2130, 1515, 700, 320, "Root prefix builder", (
        "text：pad 1→50；4096→1024", "time：PE lookup + MLP→1024", "heading：[cos,sin] 2→1024", "prefix [B,52,1024]",
    ), "code")
    box(2940, 1200, 700, 250, "Token concat", ("prefix52 + motion T", "[B,52+T,1024]"), "product")
    box(3750, 1160, 920, 360, "16× Transformer Encoder", (
        "8 heads；head dim=128", "FFN 1024→2048→1024；GELU", "post-norm；positional encoding", "输出 [B,52+T,1024]",
    ), "code")
    box(4780, 1200, 700, 250, "Slice motion tokens", ("output[:,52:]", "[B,T,1024]"), "code")
    box(5590, 1200, 700, 250, "Root output head", ("Linear 1024→5", "root_pred [B,T,5]"), "product")
    arrow(((2020, 1325), (2130, 1325)), "738D", (2075, 1305))
    arrow(((2830, 1325), (2940, 1325)), "T motion tokens", (2885, 1305))
    arrow(((2830, 1675), (2880, 1675), (2880, 1380), (2940, 1380)), "52 prefix", (2860, 1585))
    arrow(((3640, 1325), (3750, 1325)), "52+T", (3695, 1305))
    arrow(((4670, 1325), (4780, 1325)))
    arrow(((5480, 1325), (5590, 1325)), "1024→5", (5535, 1305))
    arrow(((2220, 625), (2260, 625), (2260, 1250), (2020, 1250)), "x̃", (2215, 1100))
    arrow(((2220, 700), (2300, 700), (2300, 1400), (2020, 1400)), "full mask", (2255, 1155))
    arrow(((2600, 1040), (2600, 1515)), "same raw conditions", (2600, 1100))

    # Root output split: one branch to final output, one through bridge to body.
    box(5200, 1600, 1090, 300, "5D global→4D bridge", (
        "global stats unnormalize", "Δheading×FPS；canonical-world ΔXZ×FPS；root Y", "local stats normalize → [B,T,4]",
    ), "code")
    arrow(((5940, 1450), (5940, 1600)), "root5", (5995, 1540))
    out.append('<rect x="5385" y="1905" width="720" height="58" rx="26" fill="#FFE8E6" stroke="#C24132" stroke-width="3" stroke-dasharray="13 9"/>')
    out.append(_text(5745, 1944, "TRAINING：no_grad + detach", 22, 850, "#C24132"))

    # Body path band.
    out.append('<rect x="1260" y="2050" width="5140" height="930" rx="30" fill="#FFFFFF" stroke="#5C8465" stroke-width="5"/>')
    out.append(_text(3830, 2105, "BODY PATH｜Local-root-conditioned Body Transformer", 36, 900, "#31583A"))
    box(1320, 2180, 800, 315, "Assemble Body input", (
        "root_local [B,T,4]", "+ x̃[...,body] [B,T,364]", "+ full mask [B,T,369]", "body input [B,T,737]",
    ), "product")
    box(2230, 2180, 700, 250, "Motion projection", ("Linear 737→1024", "[B,T,1024]"), "code")
    box(2230, 2500, 700, 320, "Body prefix builder", (
        "独立 text / time / heading layers", "50 + 1 + 1 = 52 tokens", "prefix [B,52,1024]", "不与 Root prefix 共享参数",
    ), "code")
    box(3040, 2180, 700, 250, "Token concat", ("prefix52 + motion T", "[B,52+T,1024]"), "product")
    box(3850, 2140, 920, 360, "16× Transformer Encoder", (
        "独立于 Root Transformer", "8 heads；head dim=128", "FFN 1024→2048→1024；GELU", "输出 [B,52+T,1024]",
    ), "code")
    box(4880, 2180, 700, 250, "Slice motion tokens", ("output[:,52:]", "[B,T,1024]"), "code")
    box(5690, 2180, 600, 250, "Body head", ("Linear 1024→364", "[B,T,364]"), "product")
    arrow(((2120, 2305), (2230, 2305)), "737D", (2175, 2285))
    arrow(((2930, 2305), (3040, 2305)), "T motion", (2985, 2285))
    arrow(((2930, 2660), (2980, 2660), (2980, 2360), (3040, 2360)), "52 prefix", (2960, 2565))
    arrow(((3740, 2305), (3850, 2305)), "52+T", (3795, 2285))
    arrow(((4770, 2305), (4880, 2305)))
    arrow(((5580, 2305), (5690, 2305)), "1024→364", (5635, 2285))
    arrow(((5200, 1770), (5050, 1770), (5050, 2020), (1480, 2020), (1480, 2180)), "detached local4", (4700, 1995), "#C24132", True)
    arrow(((1770, 775), (1770, 2010), (1650, 2010), (1650, 2180)), "x̃ body364", (1770, 1990))
    arrow(((1900, 775), (1900, 1980), (1820, 1980), (1820, 2180)), "full mask369", (1950, 1955))
    arrow(((2700, 1040), (2700, 2500)), "same raw conditions", (2700, 2050))

    # Final merge and output: right side is deliberately large and unmistakable.
    box(5300, 3070, 680, 250, "Feature concat", ("root5 + body364", "[B,T,369]"), "code")
    box(6040, 3020, 380, 350, "MODEL OUTPUT", ("clean x₀ prediction", "float [B,T,369]", "shape = input x"), "product", title_size=30, line_size=22)
    arrow(((5940, 1450), (6350, 1450), (6350, 2920), (5530, 2920), (5530, 3070)), "global root5", (6240, 2860))
    arrow(((5990, 2430), (5990, 2920), (5750, 2920), (5750, 3070)), "body364", (5985, 2860))
    arrow(((5980, 3195), (6040, 3195)), "369D", (6010, 3175))

    # Bottom equation strip.
    out.append('<rect x="75" y="3440" width="6345" height="125" rx="26" fill="#F1ECFF" stroke="#7457B5" stroke-width="4"/>')
    out.append(_text(width / 2, 3493, "完整维度主线", 25, 850, "#3B2671"))
    out.append(_text(width / 2, 3540, "[B,T,369] → concat mask → [B,T,738] → Root 1024 → global5 → bridge local4 → concat body364 + mask369 → [B,T,737] → Body 1024 → body364 → concat → [B,T,369]", 25, 750, "#3B2671"))
    out.append("</svg>")
    output.write_text("\n".join(out) + "\n", encoding="utf-8")


def render_model_io_architecture_v2(output: Path) -> None:
    """Render a readable architecture overview plus a backbone detail view."""

    width, height = 7000, 4850
    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f'<rect width="{width}" height="{height}" fill="#F8FAFC"/>',
        '<defs><marker id="smallArrow" markerWidth="10" markerHeight="10" refX="8" refY="4" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,4 L0,8 z" fill="#475569"/></marker><marker id="smallRedArrow" markerWidth="10" markerHeight="10" refX="8" refY="4" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,4 L0,8 z" fill="#C24132"/></marker><marker id="smallBlueArrow" markerWidth="10" markerHeight="10" refX="8" refY="4" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,4 L0,8 z" fill="#3973C6"/></marker></defs>',
        _text(width / 2, 90, "Kimodo TwostageDenoiser｜模型结构、输入输出与维度变化", 57, 900, "#0F172A"),
        _text(width / 2, 148, "上半图看完整模型；下半图放大 Root/Body 共用的 TransformerEncoderBlock 结构", 28, 500, "#334155"),
    ]

    def box(x: int, y: int, w: int, h: int, title: str, lines: tuple[str, ...], kind: str,
            title_size: int = 29, line_size: int = 24, dashed: bool = False) -> None:
        fill, stroke, evidence = COLORS[kind]
        dash = ' stroke-dasharray="14 10"' if dashed else ""
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="22" fill="{fill}" stroke="{stroke}" stroke-width="3"{dash}/>')
        out.append(_text(x + w / 2, y + 30, evidence, 17, 850, stroke))
        out.append(_text(x + w / 2, y + 72, title, title_size, 850, "#111827"))
        ty = y + 113
        for line in lines:
            out.append(_text(x + w / 2, ty, line, line_size, 450, "#172033"))
            ty += 39

    def flow(points: tuple[tuple[int, int], ...], label: str = "", label_xy: tuple[int, int] | None = None,
             color: str = "#475569", dashed: bool = False) -> None:
        d = "M" + " L".join(f"{x} {y}" for x, y in points)
        dash = ' stroke-dasharray="12 10"' if dashed else ""
        marker = "smallRedArrow" if color == "#C24132" else ("smallBlueArrow" if color == "#3973C6" else "smallArrow")
        out.append(f'<path d="{d}" stroke="{color}" stroke-width="4" stroke-linejoin="round" stroke-linecap="round" fill="none" marker-end="url(#{marker})"{dash}/>')
        if label and label_xy:
            lx, ly = label_xy
            label_w = max(170, len(label) * 14)
            out.append(f'<rect x="{lx-label_w/2:.0f}" y="{ly-25}" width="{label_w}" height="34" rx="15" fill="#F8FAFC"/>')
            out.append(_text(lx, ly, label, 19, 750, color))

    # Compact legend.
    legends = (("外部输入", "source"), ("公开 CODE / CONFIG", "code"), ("中间/输出张量", "product"), ("梯度边界", "boundary"))
    lx = 1450
    for label, kind in legends:
        fill, stroke, _ = COLORS[kind]
        out.append(f'<rect x="{lx}" y="185" width="930" height="54" rx="27" fill="{fill}" stroke="{stroke}" stroke-width="3"/>')
        out.append(_text(lx + 465, 221, label, 21, 850, stroke))
        lx += 970

    # ------------------------------------------------------------------
    # A. Whole model overview.  Each connector owns a separate routing lane.
    # ------------------------------------------------------------------
    out.append('<rect x="70" y="285" width="6860" height="2375" rx="32" fill="#FFFFFF" stroke="#94A3B8" stroke-width="3"/>')
    out.append(_text(210, 345, "A｜端到端模型主干：从输入到 clean-motion 输出", 39, 900, "#0F172A", anchor="start"))

    box(120, 430, 1100, 680, "动作与约束输入", (
        "noisy motion x：[B,T,369]", "它是 DDPM 加噪后的当前动作；T≤300",
        "observed_motion：[B,T,369]", "它保存 motion constraint 指定的干净真值",
        "motion_mask：[B,T,369]，True 表示该通道已知", "369D = global-root 5D + body 364D",
        "这些张量先进入右侧的 constraint overwrite",
    ), "source", line_size=25)

    box(120, 1260, 1100, 880, "文本、时间与序列输入", (
        "text_feat：[B,1,4096]", "这是离线 LLM2Vec 句向量，不是文字 token ID",
        "timesteps：[B]，指出当前 diffusion step 0…999", "first_heading_angle：[B]，给出本样本首帧目标朝向",
        "x_pad_mask：[B,T]，True=真实帧，False=padding", "text_feat_pad_mask：[B,1]",
        "Root 与 Body 读取同一组原始条件", "但各自使用独立 Linear/MLP，不共享参数",
    ), "source", line_size=25)

    box(1450, 500, 1100, 520, "1. Constraint overwrite", (
        "输入：x、observed_motion、motion_mask", "操作：mask=True 的通道用 clean observation 覆盖 noisy x",
        "x̃ = where(mask, observed_motion, x)", "输出一：x̃ [B,T,369]", "输出二：mask_float [B,T,369]",
        "目的：网络能同时看见已知真值和哪些通道可信",
    ), "code", line_size=24)

    box(1450, 1370, 1100, 570, "共享条件入口", (
        "原始条件：text、timestep、first heading", "Root block 内部生成自己的 52 个 prefix tokens",
        "Body block 内部再生成一套 52 个 prefix tokens", "52 = text slots 50 + time 1 + heading 1",
        "x_pad_mask 分别控制两个 Transformer 的 motion padding", "这里共享的是输入数值，不是投影层参数",
    ), "source", line_size=24)

    box(2820, 430, 1450, 740, "2. Global-root Transformer", (
        "动作输入：concat(x̃369, full mask369) = [B,T,738]", "每帧 Linear 738→1024，得到 T 个 motion tokens",
        "条件输入：text/time/heading → [B,52,1024] prefix", "拼接后序列：[B,52+T,1024]",
        "通过 16 层、8-head Transformer Encoder", "丢弃 52 个 prefix 输出，只保留 T 个 motion hidden",
        "Linear 1024→5，输出 root_pred [B,T,5]", "5D = smooth-root XYZ 3 + heading cos/sin 2",
    ), "code", line_size=24)

    box(4560, 500, 1050, 600, "3. Global5 → Local4 bridge", (
        "输入：归一化 root_pred [B,T,5]", "先用 global-root stats 回到物理量",
        "heading 做 wrapped 时间差分 × FPS", "canonical-world XZ 位置做时间差分 × FPS",
        "保留 root height Y，再用 local-root stats 归一化", "输出：[yaw velocity, vx, vz, root Y] = [B,T,4]",
        "训练时默认 no_grad + detach；它不是 Linear layer",
    ), "code", line_size=23)

    box(1450, 2040, 1100, 470, "4. Body 输入构造", (
        "从 x̃ 取 body slice：[B,T,364]", "拼接 bridge 给出的 root_local：[B,T,4]",
        "再拼完整原始 mask_float：[B,T,369]", "4 + 364 + 369 = 737",
        "输出 Body Transformer 输入：[B,T,737]",
    ), "product", line_size=24)

    box(2820, 1430, 1450, 760, "5. Body Transformer", (
        "动作输入：[B,T,737]；每帧 Linear 737→1024", "条件使用独立投影：text/time/heading → 52 prefix tokens",
        "拼接后同样为 [B,52+T,1024]", "通过另一套独立的 16 层、8-head Transformer Encoder",
        "Root 与 Body 的 Transformer、prefix layers 均不共享参数", "丢弃 prefix 输出，保留 [B,T,1024] motion hidden",
        "Linear 1024→364，输出 predicted_body [B,T,364]", "364D = joint pos90 + rotation180 + velocity90 + contact4",
    ), "code", line_size=24)

    box(4680, 1540, 760, 410, "6. Feature concat", (
        "输入 A：root_pred [B,T,5]", "输入 B：predicted_body [B,T,364]",
        "沿最后一维拼接", "5 + 364 = 369",
    ), "code", line_size=23)

    box(5720, 1390, 1050, 700, "MODEL OUTPUT", (
        "clean-motion prediction x̂₀", "float tensor [B,T,369]",
        "前 5D：global root XYZ + heading cos/sin", "后 364D：关节位置、旋转、速度与脚接触",
        "输出 shape 与 noisy input x 完全相同", "trainer 在模型外计算六项表示 loss + FK loss",
        "模型内部不包含 8B LLM2Vec、q_sample、optimizer 或 EMA",
    ), "product", title_size=34, line_size=25)

    # Overview flows.  Horizontal/vertical lanes are deliberately separated.
    flow(((1220, 760), (1450, 760)), "x / observed / mask", (1335, 735))
    flow(((2550, 700), (2680, 700), (2680, 760), (2820, 760)), "x̃369 + mask369", (2680, 675))
    flow(((1220, 1680), (1450, 1680)), "raw conditions", (1335, 1655), "#3973C6")
    flow(((2550, 1510), (2680, 1510), (2680, 1030), (2820, 1030)), "Root prefix", (2680, 1220), "#3973C6")
    flow(((4270, 760), (4560, 760)), "global root5", (4415, 735))
    flow(((5085, 1100), (5085, 1220), (5660, 1220), (5660, 2550), (2000, 2550), (2000, 2510)), "detached local-root4：从 bridge 单独送入 Body", (3930, 2525), "#C24132", True)
    flow(((2550, 900), (2600, 900), (2600, 2310), (2550, 2310)), "x̃ body364 + full mask369", (2600, 1325))
    flow(((2550, 1800), (2700, 1800), (2700, 1740), (2820, 1740)), "Body prefix", (2700, 1715), "#3973C6")
    flow(((2550, 2275), (2700, 2275), (2700, 1980), (2820, 1980)), "body input737", (2700, 2245))
    flow(((4270, 1820), (4680, 1820)), "body364", (4475, 1795))
    flow(((4270, 660), (4410, 660), (4410, 1470), (4920, 1470), (4920, 1540)), "root5 直接进入最终拼接", (4630, 1445))
    flow(((5440, 1745), (5720, 1745)), "clean369", (5580, 1720))

    # ------------------------------------------------------------------
    # B. Backbone zoom.  No cross-lane edges.
    # ------------------------------------------------------------------
    out.append('<rect x="70" y="2740" width="6860" height="2010" rx="32" fill="#FFFFFF" stroke="#5C8465" stroke-width="3"/>')
    out.append(_text(210, 2800, "B｜TransformerEncoderBlock 放大：每个输入如何变成 token", 39, 900, "#0F172A", anchor="start"))
    out.append(_text(210, 2845, "Root 和 Body 各有一套这样的 block；下面用 Din / Dout 表示两者不同的输入输出宽度。", 24, 500, "#475569", anchor="start"))

    box(120, 2920, 900, 310, "Text input", (
        "[B,1,4096] 句向量", "补 49 个 zero slots → [B,50,4096]", "不是把一句话重新分词成 50 tokens",
    ), "source", line_size=22)
    box(120, 3300, 900, 260, "Time input", (
        "diffusion timestep [B]", "用 step 索引 sinusoidal PE table",
    ), "source", line_size=22)
    box(120, 3630, 900, 260, "Heading input", (
        "first heading angle [B]", "先变成 [cos(angle), sin(angle)] [B,2]",
    ), "source", line_size=22)
    box(120, 4050, 900, 360, "Motion input", (
        "Root：Din=738", "Body：Din=737", "张量形状 [B,T,Din]", "每个 frame 将成为一个 motion token",
    ), "source", line_size=22)

    box(1200, 2900, 1300, 350, "Text projection", (
        "对 50 个槽位分别做 Linear 4096→1024", "输出 text tokens [B,50,1024]",
        "当前 use_text_mask=false：50 个槽位都参与 attention", "padding slot 经带 bias 的 Linear 后不保证仍为零",
    ), "code", line_size=22)
    box(1200, 3290, 1300, 300, "Timestep embedding", (
        "PE lookup 得 [B,1,1024]", "再经过 Linear1024→1024、SiLU、Linear1024→1024", "输出 1 个 time token",
    ), "code", line_size=22)
    box(1200, 3630, 1300, 280, "Heading projection", (
        "[B,2] → Linear 2→1024", "增加 token 轴 → [B,1,1024]", "输出 1 个 heading token",
    ), "code", line_size=22)
    box(1200, 4050, 1300, 360, "Motion projection", (
        "Root：Linear 738→1024", "Body：Linear 737→1024", "逐帧独立投影，不改变时间长度 T",
        "输出 motion tokens [B,T,1024]",
    ), "code", line_size=22)

    box(2780, 3090, 1050, 520, "Prefix concat", (
        "按固定顺序拼接", "text tokens：50", "+ time token：1", "+ heading token：1",
        "得到 prefix [B,52,1024]", "Root/Body 各自独立计算一次",
    ), "product", line_size=23)
    box(2780, 3990, 1050, 400, "Motion tokens", (
        "[B,T,1024]", "x_pad_mask 指出哪些 frame 有效", "padding frame 不应作为 attention key/value",
        "T≤300",
    ), "product", line_size=23)
    box(4100, 3440, 950, 590, "Sequence assembly", (
        "在 token 轴拼接 prefix + motion", "[B,52,1024] + [B,T,1024]", "→ [B,52+T,1024]",
        "T≤300，所以序列长度≤352", "加固定 sinusoidal positional encoding", "构造 src_key_padding_mask",
        "prefix 全有效；motion 使用 x_pad_mask",
    ), "product", line_size=22)
    box(5300, 3340, 1050, 800, "16 × TransformerEncoderLayer", (
        "所有 prefix 与 motion tokens 做 self-attention", "因此每个动作帧能读取文字、时间、朝向和其他帧",
        "hidden width 1024；8 attention heads", "每个 head dimension = 1024 / 8 = 128",
        "FFN：1024→2048→1024；activation=GELU", "norm_first=false，因此采用 post-norm",
        "Phase 1 attention/FFN/PE dropout=0.1", "Phase 2 dropout=0",
        "输出形状仍为 [B,52+T,1024]",
    ), "code", line_size=22)
    box(6510, 3420, 390, 650, "Output head", (
        "先移除", "前 52 个", "prefix outputs", "保留", "[B,T,1024]", "Root：→5", "Body：→364",
    ), "product", title_size=26, line_size=20)

    # Backbone flows: thin, short and never through another node.
    flow(((1020, 3070), (1200, 3070)))
    flow(((1020, 3430), (1200, 3430)))
    flow(((1020, 3760), (1200, 3760)))
    flow(((1020, 4230), (1200, 4230)))
    flow(((2500, 3070), (2640, 3070), (2640, 3200), (2780, 3200)), "50 tokens", (2630, 3045))
    flow(((2500, 3440), (2640, 3440), (2640, 3345), (2780, 3345)), "1 token", (2630, 3415))
    flow(((2500, 3765), (2700, 3765), (2700, 3490), (2780, 3490)), "1 token", (2690, 3740))
    flow(((2500, 4230), (2780, 4230)), "T tokens", (2640, 4205))
    flow(((3830, 3350), (3970, 3350), (3970, 3650), (4100, 3650)), "prefix52", (3955, 3325))
    flow(((3830, 4190), (3970, 4190), (3970, 3850), (4100, 3850)), "motion T", (3955, 4165))
    flow(((5050, 3735), (5300, 3735)), "[B,52+T,1024]", (5175, 3710))
    flow(((6350, 3735), (6510, 3735)), "slice + Linear", (6430, 3710))

    out.append('<rect x="120" y="4515" width="6460" height="145" rx="24" fill="#F1ECFF" stroke="#7457B5" stroke-width="3"/>')
    out.append(_text(width / 2, 4567, "两套 block 的唯一结构差别", 25, 850, "#3B2671"))
    out.append(_text(width / 2, 4612, "Root：Din=738、Dout=5　｜　Body：Din=737、Dout=364　｜　其余 latent=1024、prefix=52、layers=16、heads=8、FFN=2048 相同，但参数完全不共享", 24, 700, "#3B2671"))
    out.append("</svg>")
    output.write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    render(
        "Kimodo / SOMA-SEED 数据处理全流程｜FRESH 重建路线",
        "公开 BVH + annotations → canonical NPZ → manifest → LLM2Vec → stats → online 369D → trainer batch",
        DATA_STAGES,
        here / "data_pipeline_technical_share.svg",
        "未复现边界：Qwen3-32B paraphrase 与 cross-motion stitching + diffusion transition 没有静默混入当前 public baseline。",
    )
    render(
        "当前机器实际路线｜验证 legacy 资产并封装 portable bundle",
        "不重新解析 BVH、不重新生成 NPZ、不重新运行 8B LLM2Vec；验证、重绑路径、全量校验后原子发布",
        ADOPTION_STAGES,
        here / "data_bundle_adoption_technical_share.svg",
        "此路线证明 legacy payload 符合当前数据合同；它不等于 fresh converter 与 LLM2Vec 已对全量数据重新执行。",
    )
    render_readable_timeline(
        here / "two_stage_training_technical_share.svg",
        title="Kimodo 两阶段训练｜从 DataLoader 到参数更新",
        subtitle="课程期决定训练策略；模型第 1/2 段在每次 forward 内连续执行",
        stages=TRAIN_STAGES,
        conclusion=(
            "DataLoader 提供 clean target；Trainer 在线生成噪声、文本 dropout 与动作约束。",
            "模型第 1 段预测 root5，经 bridge 得 local4，再由模型第 2 段预测 body364。",
        ),
    )
    render_readable_timeline(
        here / "model_architecture_technical_share.svg",
        title="Kimodo TwostageDenoiser｜模型内部逐张量结构",
        subtitle="从四类模型输入到 Root、Bridge、Body 与 clean-motion 输出",
        stages=MODEL_STAGES,
        conclusion=(
            "Root 看完整 369D 动作与完整 mask；Body 读取 detached local4 和 body364。",
            "最终拼接 root5 + body364，输出与 noisy-motion 同 shape 的 clean x0 prediction。",
        ),
    )
