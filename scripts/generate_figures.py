"""
Generate professional figures for README.md
- Pipeline architecture diagram
- Performance improvement chart
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

# ─── Common Style ─────────────────────────────────────────────
plt.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 9,
    "axes.unicode_minus": False,
})

COLORS = {
    "input":    "#3B82F6",
    "convert":  "#06B6D4",
    "detect":   "#F59E0B",
    "post":     "#EF4444",
    "ocr":      "#10B981",
    "order":    "#8B5CF6",
    "output":   "#6366F1",
    "bg":       "#FFFFFF",
    "card_bg":  "#F8FAFC",
    "text":     "#1E293B",
    "muted":    "#64748B",
    "accent":   "#2563EB",
    "border":   "#E2E8F0",
}


def _is_dark(hex_color):
    h = hex_color.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16)
    return (0.299 * r + 0.587 * g + 0.114 * b) < 150


def draw_rounded_box(ax, x, y, w, h, color, lines, *,
                     fontsize=9, radius=0.015, alpha=1.0, edge_color="none",
                     text_color=None, bold_idx=None, line_spacing=1.3):
    tc = text_color or ("#FFFFFF" if _is_dark(color) else "#1E293B")
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=color, edgecolor=edge_color, alpha=alpha,
        linewidth=0.8, zorder=2,
    )
    ax.add_patch(box)
    total_h = len(lines) * fontsize * 0.0095 * line_spacing
    start_y = y + h / 2 + total_h / 2 - fontsize * 0.0095 * 0.5
    for i, line in enumerate(lines):
        yy = start_y - i * fontsize * 0.0095 * line_spacing
        weight = "bold" if (bold_idx is not None and i in (bold_idx if isinstance(bold_idx, (list, tuple)) else [bold_idx])) else "normal"
        ax.text(x + w / 2, yy, line, ha="center", va="center",
                fontsize=fontsize, color=tc, weight=weight, zorder=3)


def draw_arrow(ax, x0, y0, x1, y1, *, color="#94A3B8", lw=1.5, style="-|>"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                mutation_scale=14), zorder=4)


# ═════════════════════════════════════════════════════════════════
#  FIGURE 1: Pipeline Architecture
# ═════════════════════════════════════════════════════════════════
def generate_pipeline_figure():
    fig, ax = plt.subplots(figsize=(20, 12), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(COLORS["bg"])

    # ── Title
    ax.text(0.5, 0.97, "System Pipeline Architecture",
            ha="center", va="top", fontsize=20, weight="bold", color=COLORS["text"])
    ax.text(0.5, 0.94, "Visually-rich Document Understanding  |  5-Stage Multi-modal Pipeline",
            ha="center", va="top", fontsize=11, color=COLORS["muted"])

    # ── Main Flow (Row 1) ───────────────────────────────────────
    ROW_Y = 0.74
    BOX_W = 0.125
    BOX_H = 0.12
    GAP = 0.018
    START_X = 0.045
    xs = [START_X + i * (BOX_W + GAP) for i in range(6)]

    stages = [
        (COLORS["input"],   ["Input", "Document"],         "1"),
        (COLORS["convert"], ["Document", "Conversion"],    "2"),
        (COLORS["detect"],  ["Layout", "Detection"],       "3"),
        (COLORS["post"],    ["Post-", "Processing"],       "4"),
        (COLORS["order"],   ["Reading", "Order"],          "5"),
        (COLORS["ocr"],     ["OCR + Text", "Extraction"],  "6"),
    ]

    for i, (color, lines, num) in enumerate(stages):
        draw_rounded_box(ax, xs[i], ROW_Y, BOX_W, BOX_H, color, lines,
                         fontsize=11, bold_idx=0)
        # Stage badge
        cx = xs[i] + BOX_W / 2
        cy = ROW_Y + BOX_H + 0.025
        circle = plt.Circle((cx, cy), 0.014, color=color, zorder=5)
        ax.add_patch(circle)
        ax.text(cx, cy, num, ha="center", va="center",
                fontsize=9, color="white", weight="bold", zorder=6)

    # Arrows between stages
    for i in range(5):
        draw_arrow(ax, xs[i] + BOX_W + 0.003, ROW_Y + BOX_H / 2,
                   xs[i+1] - 0.003, ROW_Y + BOX_H / 2)

    # Output box
    OUT_X = xs[5] + BOX_W + GAP + 0.01
    draw_rounded_box(ax, OUT_X, ROW_Y + 0.015, 0.09, BOX_H - 0.03,
                     "#1E293B", ["CSV", "Output"], fontsize=10, bold_idx=0)
    draw_arrow(ax, xs[5] + BOX_W + 0.003, ROW_Y + BOX_H / 2,
               OUT_X - 0.003, ROW_Y + BOX_H / 2)

    # ── Detail Cards (Row 2) ─────────────────────────────────────
    CARD_Y = 0.34
    CARD_H = 0.33
    CARD_W = BOX_W + 0.01

    def detail_card(x, y, w, h, title, items, accent_color):
        # Card background
        draw_rounded_box(ax, x, y, w, h, "#F8FAFC", [],
                         edge_color=COLORS["border"], alpha=0.95, radius=0.012)
        # Colored top bar
        bar = FancyBboxPatch(
            (x, y + h - 0.035), w, 0.035,
            boxstyle="round,pad=0,rounding_size=0.012",
            facecolor=accent_color, edgecolor="none", alpha=0.9, zorder=2)
        ax.add_patch(bar)
        ax.text(x + w / 2, y + h - 0.017, title,
                ha="center", va="center", fontsize=8.5, color="white",
                weight="bold", zorder=3)
        # Items
        item_h = 0.038
        item_gap = 0.008
        start_y = y + h - 0.055
        for j, (label, sublabel) in enumerate(items):
            iy = start_y - j * (item_h + item_gap)
            draw_rounded_box(ax, x + 0.006, iy, w - 0.012, item_h,
                             accent_color, [label, sublabel] if sublabel else [label],
                             fontsize=6.8, alpha=0.15 + 0.1 * (j % 2),
                             text_color=COLORS["text"], line_spacing=1.1)

    # Card connectors
    for i in [1, 2, 3, 4, 5]:
        cx = xs[i] + BOX_W / 2
        draw_arrow(ax, cx, ROW_Y - 0.005, cx, CARD_Y + CARD_H + 0.005,
                   color="#CBD5E1", lw=1.0, style="-|>")

    # Stage 2: Document Conversion
    detail_card(xs[1] - 0.005, CARD_Y, CARD_W, CARD_H, "CONVERSION", [
        ("PDF  ->  pdf2image", "Poppler backend"),
        ("PPTX -> LibreOffice", "-> PDF -> PNG"),
        ("JPG / PNG", "Direct PIL load"),
        ("800 DPI rendering", "High-res output"),
        ("Letterbox resize", "1280 x 1280 px"),
    ], COLORS["convert"])

    # Stage 3: Layout Detection
    detail_card(xs[2] - 0.005, CARD_Y, CARD_W, CARD_H, "DUAL-MODEL", [
        ("Portrait / PDF", "YOLOv12-DocLayNet"),
        ("Landscape / PPTX", "Fine-tuned V5.pt"),
        ("6-class detection", "title/sub/text/tbl/img/eq"),
        ("Per-class threshold", "Optuna-optimized"),
        ("imgsz = 1024", "YOLOv12-Large"),
    ], COLORS["detect"])

    # Stage 4: Post-Processing
    detail_card(xs[3] - 0.005, CARD_Y, CARD_W, CARD_H, "4 RULES", [
        ("R1. NMS", "IoU threshold = 0.5"),
        ("R2. Sub -> Title", "conf < 0.80 promote"),
        ("R3. Single Title", "Top-left title only"),
        ("R4. Caption Filter", "Fig/Table pattern remove"),
        ("Duplicate removal", "Confidence-sorted"),
    ], COLORS["post"])

    # Stage 5: Reading Order
    detail_card(xs[4] - 0.005, CARD_Y, CARD_W, CARD_H, "3 STRATEGIES", [
        ("Book-style", "Single column: T->B, L->R"),
        ("Newspaper-style", "Multi-col KMeans detect"),
        ("Poster Scan-mode", "Subtitle-row partitioning"),
        ("Column detection", "k=2,3 with gap validation"),
        ("Numeric sort", "Auto prefix ordering"),
    ], COLORS["order"])

    # Stage 6: OCR
    detail_card(xs[5] - 0.005, CARD_Y, CARD_W, CARD_H, "EASYOCR", [
        ("CRAFT detector", "Text region detection"),
        ("Korean model", "korean_g2.pth"),
        ("English model", "english_g2.pth"),
        ("10-step cleaning", "NFKC / HTML / bullets"),
        ("GPU accelerated", "CUDA auto-detect"),
    ], COLORS["ocr"])

    # ── Fine-tuning Pipeline (Row 3) ─────────────────────────────
    FT_Y = 0.06
    FT_H = 0.07
    FT_BG_Y = FT_Y - 0.02
    FT_BG_H = FT_H + 0.12

    # Background
    draw_rounded_box(ax, 0.03, FT_BG_Y, 0.82, FT_BG_H, "#EFF6FF", [],
                     edge_color="#BFDBFE", radius=0.015, alpha=0.6)
    ax.text(0.05, FT_BG_Y + FT_BG_H - 0.015, "Fine-tuning Pipeline",
            fontsize=10, weight="bold", color="#1D4ED8")

    ft_stages = [
        (0.05,  0.12, "#93C5FD", ["NeurIPS Posters", "(100 img, 6 cls)"]),
        (0.20,  0.11, "#60A5FA", ["Train / Val Split", "(90% / 10%)"]),
        (0.34,  0.13, "#3B82F6", ["YOLOv12-L", "Fine-tuning"]),
        (0.51,  0.14, "#2563EB", ["imgsz=1024, bs=16", "cosine LR, p=50"]),
        (0.69,  0.12, "#1D4ED8", ["V5.pt", "mAP50 = 0.852"]),
    ]

    for i, (fx, fw, fc, fl) in enumerate(ft_stages):
        draw_rounded_box(ax, fx, FT_Y, fw, FT_H, fc, fl, fontsize=8, bold_idx=0)
        if i < len(ft_stages) - 1:
            nx = ft_stages[i+1][0]
            draw_arrow(ax, fx + fw + 0.003, FT_Y + FT_H / 2,
                       nx - 0.003, FT_Y + FT_H / 2, color="#3B82F6", lw=1.2)

    # Dashed arrow from V5.pt to Detection stage
    ax.annotate("", xy=(xs[2] + BOX_W * 0.7, ROW_Y - 0.003),
                xytext=(0.75, FT_Y + FT_H + 0.01),
                arrowprops=dict(arrowstyle="-|>", color="#1D4ED8",
                                lw=1.5, linestyle=(0, (5, 3))),
                zorder=4)

    # ── 6-class legend ────────────────────────────────────────────
    cls_items = [
        ("Title", "#EF4444"), ("Subtitle", "#F59E0B"), ("Text", "#FDE047"),
        ("Image", "#4ADE80"), ("Table", "#60A5FA"), ("Equation", "#C084FC"),
    ]
    leg_x = 0.87
    leg_y = 0.50
    draw_rounded_box(ax, leg_x - 0.01, leg_y - 0.01, 0.12, 0.24,
                     "#F8FAFC", [], edge_color=COLORS["border"], radius=0.01)
    ax.text(leg_x + 0.05, leg_y + 0.21, "6 Classes",
            ha="center", fontsize=9, weight="bold", color=COLORS["text"])
    for i, (name, color) in enumerate(cls_items):
        yy = leg_y + 0.17 - i * 0.033
        ax.add_patch(FancyBboxPatch(
            (leg_x, yy - 0.009), 0.018, 0.018,
            boxstyle="round,pad=0,rounding_size=0.004",
            facecolor=color, edgecolor="none", zorder=2))
        ax.text(leg_x + 0.025, yy, name, fontsize=8, va="center",
                color=COLORS["text"])

    out = ASSETS_DIR / "pipeline_architecture.png"
    fig.savefig(out, dpi=180, bbox_inches="tight",
                facecolor=COLORS["bg"], pad_inches=0.15)
    plt.close(fig)
    print(f"Saved: {out}")


# ═════════════════════════════════════════════════════════════════
#  FIGURE 2: Performance Improvement Chart
# ═════════════════════════════════════════════════════════════════
def generate_performance_figure():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=180)
    fig.patch.set_facecolor(COLORS["bg"])

    # ── Left: Module-wise Score Improvement (horizontal bar) ──
    stages = [
        "Baseline",
        "+ Rule-based\n  Post-processing",
        "+ Fine-tuning\n  (V5.pt)",
        "+ Optuna\n  Optimization",
        "+ Reading Order\n  Algorithms",
        "+ OCR\n  (800 DPI)",
    ]
    scores = [0.1168, 0.245, 0.436, 0.443, 0.457, 0.4577]
    colors_bar = ["#94A3B8", "#EF4444", "#F59E0B", "#8B5CF6", "#3B82F6", "#10B981"]

    y_pos = np.arange(len(stages))
    bars = ax1.barh(y_pos, scores, color=colors_bar, height=0.6, edgecolor="white", linewidth=0.5)

    for bar, score in zip(bars, scores):
        ax1.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height() / 2,
                 f"{score:.4f}", va="center", fontsize=10, weight="bold", color=COLORS["text"])

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(stages, fontsize=9)
    ax1.set_xlabel("Public Score", fontsize=11, color=COLORS["text"])
    ax1.set_title("Module-wise Performance Improvement", fontsize=13, weight="bold",
                  color=COLORS["text"], pad=15)
    ax1.set_xlim(0, 0.55)
    ax1.invert_yaxis()
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.tick_params(left=False)

    # +292% annotation
    ax1.annotate("+292%", xy=(0.4577, 5), xytext=(0.50, 2.5),
                 fontsize=16, weight="bold", color="#DC2626",
                 arrowprops=dict(arrowstyle="-|>", color="#DC2626", lw=2),
                 ha="center", va="center")

    # ── Right: Training Curve ──
    epochs = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 61]
    map50   = [0.132, 0.580, 0.789, 0.830, 0.849, 0.845, 0.849, 0.850, 0.856, 0.850, 0.849, 0.851, 0.852]
    map5095 = [0.100, 0.400, 0.627, 0.670, 0.692, 0.680, 0.682, 0.690, 0.699, 0.690, 0.694, 0.698, 0.701]
    box_loss = [1.128, 0.750, 0.595, 0.540, 0.506, 0.485, 0.468, 0.458, 0.450, 0.442, 0.434, 0.428, 0.422]

    ax2_twin = ax2.twinx()

    l1, = ax2.plot(epochs, map50, "o-", color="#3B82F6", linewidth=2.5, markersize=5, label="mAP50", zorder=3)
    l2, = ax2.plot(epochs, map5095, "s-", color="#10B981", linewidth=2.5, markersize=5, label="mAP50-95", zorder=3)
    l3, = ax2_twin.plot(epochs, box_loss, "^--", color="#EF4444", linewidth=1.5, markersize=4, alpha=0.7, label="Box Loss")

    ax2.set_xlabel("Epoch", fontsize=11, color=COLORS["text"])
    ax2.set_ylabel("mAP Score", fontsize=11, color="#3B82F6")
    ax2_twin.set_ylabel("Box Loss", fontsize=11, color="#EF4444")
    ax2.set_title("Fine-tuning Training Curve", fontsize=13, weight="bold",
                  color=COLORS["text"], pad=15)
    ax2.set_ylim(0, 1.0)
    ax2_twin.set_ylim(0, 1.3)
    ax2.set_xlim(0, 65)

    # Final performance annotation
    ax2.annotate(f"mAP50 = 0.852", xy=(61, 0.852), xytext=(45, 0.55),
                 fontsize=11, weight="bold", color="#1D4ED8",
                 arrowprops=dict(arrowstyle="-|>", color="#1D4ED8", lw=1.5),
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#EFF6FF", edgecolor="#BFDBFE"))

    # Early stop line
    ax2.axvline(x=61, color="#94A3B8", linestyle=":", linewidth=1, alpha=0.7)
    ax2.text(62, 0.1, "Early\nStop", fontsize=8, color="#94A3B8", ha="left")

    # Convergence region
    ax2.axvspan(30, 65, alpha=0.05, color="#3B82F6")
    ax2.text(47, 0.15, "Convergence", fontsize=8, color="#94A3B8", ha="center", style="italic")

    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="lower right", fontsize=9, framealpha=0.9)

    ax2.spines["top"].set_visible(False)
    ax2_twin.spines["top"].set_visible(False)
    ax2.grid(axis="y", alpha=0.2)

    plt.tight_layout(pad=2.0)
    out = ASSETS_DIR / "performance_results.png"
    fig.savefig(out, dpi=180, bbox_inches="tight",
                facecolor=COLORS["bg"], pad_inches=0.2)
    plt.close(fig)
    print(f"Saved: {out}")


# ═════════════════════════════════════════════════════════════════
#  FIGURE 3: Reading Order Strategies
# ═════════════════════════════════════════════════════════════════
def generate_reading_order_figure():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=180)
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle("3 Reading Order Strategies", fontsize=16, weight="bold",
                 color=COLORS["text"], y=0.98)

    strategies = [
        {
            "title": "Book-style",
            "subtitle": "Single Column Documents",
            "color": "#3B82F6",
            "desc": [
                "1. Sort all items by (y, x)",
                "2. Group into lines by y-tolerance",
                "3. Within line: left -> right",
                "4. Lines: top -> bottom",
            ],
            "applicable": "PDF, Report, Single-column",
            "improvement": "+21.4%",
        },
        {
            "title": "Newspaper-style",
            "subtitle": "Multi-column Documents",
            "color": "#F59E0B",
            "desc": [
                "1. Detect text spans (>=65% width)",
                "2. Spans divide page into regions",
                "3. KMeans column detection (k=2,3)",
                "4. Read column-by-column, T->B",
            ],
            "applicable": "Paper, Academic article",
            "improvement": "+57.1%",
        },
        {
            "title": "Poster Scan-mode",
            "subtitle": "Landscape Documents",
            "color": "#8B5CF6",
            "desc": [
                "1. Group subtitles into rows",
                "2. Scan top -> bottom",
                "3. Subtitle: define vertical strips",
                "4. Read each strip (multi-column)",
            ],
            "applicable": "PPTX, Poster, Landscape",
            "improvement": "+128.6%",
        },
    ]

    for ax, s in zip(axes, strategies):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Card background
        draw_rounded_box(ax, 0.05, 0.05, 0.9, 0.88, "#F8FAFC", [],
                         edge_color=COLORS["border"], radius=0.03)

        # Header bar
        header = FancyBboxPatch(
            (0.05, 0.78), 0.9, 0.15,
            boxstyle="round,pad=0,rounding_size=0.03",
            facecolor=s["color"], edgecolor="none", alpha=0.95, zorder=2)
        ax.add_patch(header)

        ax.text(0.5, 0.88, s["title"], ha="center", va="center",
                fontsize=16, weight="bold", color="white", zorder=3)
        ax.text(0.5, 0.815, s["subtitle"], ha="center", va="center",
                fontsize=10, color="white", alpha=0.9, zorder=3)

        # Steps
        for i, step in enumerate(s["desc"]):
            y = 0.68 - i * 0.12
            draw_rounded_box(ax, 0.1, y - 0.02, 0.8, 0.09, s["color"],
                             [step], fontsize=10, alpha=0.1,
                             text_color=COLORS["text"], line_spacing=1.0)

        # Applicable
        ax.text(0.5, 0.19, s["applicable"], ha="center", va="center",
                fontsize=9, color=COLORS["muted"], style="italic")

        # Improvement badge
        draw_rounded_box(ax, 0.3, 0.06, 0.4, 0.09, s["color"],
                         [s["improvement"]], fontsize=14, bold_idx=0)

    plt.tight_layout(pad=1.5)
    out = ASSETS_DIR / "reading_order_strategies.png"
    fig.savefig(out, dpi=180, bbox_inches="tight",
                facecolor=COLORS["bg"], pad_inches=0.15)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    generate_pipeline_figure()
    generate_performance_figure()
    generate_reading_order_figure()
    print("All figures generated!")
