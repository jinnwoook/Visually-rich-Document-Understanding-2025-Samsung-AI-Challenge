"""
Visually-rich Document Understanding – Method Pipeline Figure
논문 방법론 섹션용 파이프라인 그림 생성 스크립트
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Font ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "NanumGothic",
    "font.size": 9,
    "axes.unicode_minus": False,
})

# ── Color Palette (academic, muted) ──────────────────────────────
C = {
    "input":      "#4A90D9",   # blue
    "convert":    "#5BA8C8",   # teal
    "detect":     "#E8913A",   # orange
    "post":       "#D4A03E",   # gold
    "ocr":        "#6BAF6B",   # green
    "order":      "#9B6BB0",   # purple
    "output":     "#C75B5B",   # red
    "branch_a":   "#E8913A",   # orange (portrait)
    "branch_b":   "#D06040",   # darker orange (landscape)
    "sub":        "#F0F0F0",   # light grey for sub-boxes
    "arrow":      "#444444",
    "bg":         "#FFFFFF",
    "text_dark":  "#1A1A1A",
    "text_light": "#FFFFFF",
}


def rounded_box(ax, x, y, w, h, color, label_lines, *, fontsize=8.5,
                radius=0.02, text_color=None, bold_first=True, alpha=1.0):
    """Draw a rounded rectangle with multi-line centered text."""
    tc = text_color or ("#FFFFFF" if _is_dark(color) else "#1A1A1A")
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=color, edgecolor="none", alpha=alpha,
        transform=ax.transData, zorder=2,
    )
    ax.add_patch(box)
    # text
    for i, line in enumerate(label_lines):
        yy = y + h / 2 + (len(label_lines) - 1) * fontsize * 0.006 - i * fontsize * 0.012
        weight = "bold" if (i == 0 and bold_first) else "normal"
        ax.text(x + w / 2, yy, line, ha="center", va="center",
                fontsize=fontsize, color=tc, weight=weight, zorder=3)
    return box


def arrow(ax, x0, y0, x1, y1, *, color=C["arrow"], lw=1.2, head=0.012):
    """Simple arrow between two points."""
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=12),
                zorder=4)


def bracket_arrow(ax, x0, y0, x_mid, x1, y1, *, color=C["arrow"], lw=1.0):
    """L-shaped connector: down then right."""
    ax.plot([x0, x0, x1], [y0, y1, y1], color=color, lw=lw, zorder=1)
    ax.annotate("", xy=(x1 + 0.002, y1), xytext=(x1 - 0.005, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=10), zorder=4)


def _is_dark(hex_color):
    h = hex_color.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16)
    return (0.299 * r + 0.587 * g + 0.114 * b) < 150


# =====================================================================
#  MAIN FIGURE
# =====================================================================
fig, ax = plt.subplots(figsize=(18, 10), dpi=200)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor(C["bg"])

# ── Title ─────────────────────────────────────────────────────────
ax.text(0.5, 0.97, "Proposed Method Pipeline", ha="center", va="top",
        fontsize=16, weight="bold", color=C["text_dark"])
ax.text(0.5, 0.945, "Visually-rich Document Understanding for Multi-format Documents",
        ha="center", va="top", fontsize=10, color="#666666")

# ── Layout constants ──────────────────────────────────────────────
ROW1_Y = 0.73          # main pipeline row
BOX_H = 0.14
BOX_W = 0.105
GAP   = 0.024
START_X = 0.035

xs = [START_X + i * (BOX_W + GAP) for i in range(7)]

# =====================================================================
#  ROW 1 – Main Pipeline Boxes
# =====================================================================
stages = [
    (C["input"],   ["Input", "Document", "", "PDF / PPTX", "JPG / PNG"]),
    (C["convert"], ["Document", "Conversion", "", "pdf2image", "@ 800 DPI"]),
    (C["detect"],  ["Layout", "Detection", "", "Dual-Model", "YOLOv12-L"]),
    (C["post"],    ["Post-", "Processing", "", "NMS + Class", "Refinement"]),
    (C["ocr"],     ["OCR & Text", "Extraction", "", "EasyOCR", "(ko + en)"]),
    (C["order"],   ["Reading", "Order", "", "Adaptive", "Algorithm"]),
    (C["output"],  ["Output", "(CSV)", "", "ID / class / conf", "order / text / bbox"]),
]

for i, (color, lines) in enumerate(stages):
    rounded_box(ax, xs[i], ROW1_Y, BOX_W, BOX_H, color, lines, fontsize=8.5)

# arrows between main boxes
for i in range(6):
    arrow(ax, xs[i] + BOX_W, ROW1_Y + BOX_H / 2,
          xs[i + 1], ROW1_Y + BOX_H / 2)

# =====================================================================
#  ROW 2 – Detail panels below each stage
# =====================================================================
DET_Y = 0.38
DET_H = 0.30
DET_W = BOX_W

# --- Stage 1 detail: Document Conversion ----
x = xs[1]
rounded_box(ax, x - 0.005, DET_Y, DET_W + 0.01, DET_H, "#E8F4F8",
            [], radius=0.015, alpha=0.7)
sub_h = 0.055
sub_gap = 0.012
items_conv = [
    ("PDF → pdf2image", "#5BA8C8"),
    ("PPTX → LibreOffice\n→ PDF → PNG", "#5BA8C8"),
    ("JPG/PNG → Direct", "#5BA8C8"),
    ("Output: PIL Image\n+ Metadata", "#3D8B9E"),
]
for j, (txt, col) in enumerate(items_conv):
    yy = DET_Y + DET_H - 0.03 - j * (sub_h + sub_gap)
    rounded_box(ax, x + 0.003, yy - sub_h, DET_W - 0.006, sub_h,
                col, txt.split("\n"), fontsize=6.5, alpha=0.85)
# vertical arrow from main box
arrow(ax, xs[1] + BOX_W / 2, ROW1_Y,
      xs[1] + BOX_W / 2, DET_Y + DET_H, color="#888888", lw=0.8)

# --- Stage 2 detail: Dual-Model Detection ----
x = xs[2]
rounded_box(ax, x - 0.005, DET_Y, DET_W + 0.01, DET_H, "#FFF3E0",
            [], radius=0.015, alpha=0.7)
# branch label
ax.text(x + DET_W / 2, DET_Y + DET_H - 0.015, "Document Type Check",
        ha="center", fontsize=7, weight="bold", color="#333")
# two branches
bw = (DET_W - 0.02) / 2
bh = 0.10
bx_l = x + 0.003
bx_r = x + DET_W / 2 + 0.002
by = DET_Y + DET_H - 0.16

rounded_box(ax, bx_l, by, bw, bh, C["branch_a"],
            ["Portrait / PDF", "", "DocLayNet", "(base)"], fontsize=6.5)
rounded_box(ax, bx_r, by, bw, bh, C["branch_b"],
            ["Landscape / PPTX", "", "V5.pt", "(fine-tuned)"], fontsize=6.5)

# confidence boxes
cy = DET_Y + 0.01
ch = 0.075
rounded_box(ax, bx_l, cy, bw, ch, "#F5DEB3",
            ["Conf: Title=0.15", "Text=0.10"], fontsize=5.8,
            text_color="#333", bold_first=False)
rounded_box(ax, bx_r, cy, bw, ch, "#F5DEB3",
            ["Conf: Title=0.45", "Image=0.22"], fontsize=5.8,
            text_color="#333", bold_first=False)

arrow(ax, xs[2] + BOX_W / 2, ROW1_Y,
      xs[2] + BOX_W / 2, DET_Y + DET_H, color="#888888", lw=0.8)

# --- Stage 3 detail: Post-Processing ----
x = xs[3]
rounded_box(ax, x - 0.005, DET_Y, DET_W + 0.01, DET_H, "#FFF8E1",
            [], radius=0.015, alpha=0.7)
items_post = [
    ("① NMS\n(IoU=0.5)", "#D4A03E"),
    ("② Subtitle→Title\nPromotion", "#C9963A"),
    ("③ Enforce Single\nTop Title", "#BE8C36"),
    ("④ Caption Text\nFiltering", "#B38232"),
]
for j, (txt, col) in enumerate(items_post):
    yy = DET_Y + DET_H - 0.025 - j * (sub_h + sub_gap * 0.7)
    rounded_box(ax, x + 0.003, yy - sub_h, DET_W - 0.006, sub_h,
                col, txt.split("\n"), fontsize=6.2, alpha=0.9)
# flow arrows between sub-steps
for j in range(3):
    yy_from = DET_Y + DET_H - 0.025 - j * (sub_h + sub_gap * 0.7) - sub_h
    yy_to = DET_Y + DET_H - 0.025 - (j + 1) * (sub_h + sub_gap * 0.7)
    arrow(ax, x + DET_W / 2, yy_from, x + DET_W / 2, yy_to,
          color="#999", lw=0.6, head=0.008)

arrow(ax, xs[3] + BOX_W / 2, ROW1_Y,
      xs[3] + BOX_W / 2, DET_Y + DET_H, color="#888888", lw=0.8)

# --- Stage 4 detail: OCR ----
x = xs[4]
rounded_box(ax, x - 0.005, DET_Y, DET_W + 0.01, DET_H, "#E8F5E9",
            [], radius=0.015, alpha=0.7)
items_ocr = [
    ("Region Crop\n(bbox → crop)", "#6BAF6B"),
    ("EasyOCR\nKorean + English", "#5A9E5A"),
    ("10-step Text\nCleaning", "#4A8E4A"),
    ("NFKC · Tag removal\nBullet · Space norm", "#3A7E3A"),
]
for j, (txt, col) in enumerate(items_ocr):
    yy = DET_Y + DET_H - 0.025 - j * (sub_h + sub_gap * 0.7)
    rounded_box(ax, x + 0.003, yy - sub_h, DET_W - 0.006, sub_h,
                col, txt.split("\n"), fontsize=6.2, alpha=0.9)
for j in range(3):
    yy_from = DET_Y + DET_H - 0.025 - j * (sub_h + sub_gap * 0.7) - sub_h
    yy_to = DET_Y + DET_H - 0.025 - (j + 1) * (sub_h + sub_gap * 0.7)
    arrow(ax, x + DET_W / 2, yy_from, x + DET_W / 2, yy_to,
          color="#999", lw=0.6, head=0.008)

arrow(ax, xs[4] + BOX_W / 2, ROW1_Y,
      xs[4] + BOX_W / 2, DET_Y + DET_H, color="#888888", lw=0.8)

# --- Stage 5 detail: Reading Order ----
x = xs[5]
rounded_box(ax, x - 0.005, DET_Y, DET_W + 0.01, DET_H, "#F3E5F5",
            [], radius=0.015, alpha=0.7)
ax.text(x + DET_W / 2, DET_Y + DET_H - 0.015, "Adaptive Selection",
        ha="center", fontsize=7, weight="bold", color="#333")

bw2 = (DET_W - 0.02) / 2
bh2 = 0.19
bx_l2 = x + 0.003
bx_r2 = x + DET_W / 2 + 0.002
by2 = DET_Y + 0.01

# Default mode
rounded_box(ax, bx_l2, by2, bw2, bh2, "#9B6BB0",
            ["Default Mode", "(Portrait/PDF)", "",
             "Span Detection", "Multi-column", "KMeans (k=2,3)",
             "Top→Bot, L→R"],
            fontsize=5.5)

# Scan mode
rounded_box(ax, bx_r2, by2, bw2, bh2, "#7B4B90",
            ["Scan Mode", "(Landscape/PPTX)", "",
             "Subtitle Row", "Grouping", "Vertical Strip",
             "Numeric Sort"],
            fontsize=5.5)

arrow(ax, xs[5] + BOX_W / 2, ROW1_Y,
      xs[5] + BOX_W / 2, DET_Y + DET_H, color="#888888", lw=0.8)

# =====================================================================
#  ROW 3 – Fine-tuning pipeline (bottom)
# =====================================================================
FT_Y = 0.05
FT_H = 0.065
ft_boxes = [
    (0.04, 0.11, "#BBDEFB", ["NeurIPS Posters", "(100 images, 6 cls)"]),
    (0.18, 0.11, "#90CAF9", ["Train/Val Split", "(90% / 10%)"]),
    (0.32, 0.13, "#64B5F6", ["Fine-tuning", "YOLOv12-L on DocLayNet"]),
    (0.49, 0.14, "#42A5F5", ["Training Config", "imgsz=1024, batch=16", "cos_lr, patience=50"]),
    (0.67, 0.12, "#2196F3", ["V5.pt (Fine-tuned)", "mAP50=0.852"]),
]
ax.text(0.02, FT_Y + FT_H + 0.07, "Fine-tuning Pipeline",
        fontsize=9, weight="bold", color="#1565C0")
# light bg
rounded_box(ax, 0.02, FT_Y - 0.01, 0.78, FT_H + 0.095, "#E3F2FD",
            [], radius=0.015, alpha=0.4)

for i, (fx, fw, fc, fl) in enumerate(ft_boxes):
    rounded_box(ax, fx, FT_Y, fw, FT_H, fc, fl, fontsize=6.8)
    if i < len(ft_boxes) - 1:
        nx, nw = ft_boxes[i + 1][0], ft_boxes[i + 1][1]
        arrow(ax, fx + fw, FT_Y + FT_H / 2,
              nx, FT_Y + FT_H / 2, color="#1565C0", lw=1.0)

# dashed arrow from V5.pt up to detection stage
ax.annotate("", xy=(xs[2] + BOX_W * 0.75, ROW1_Y),
            xytext=(0.67 + 0.06, FT_Y + FT_H),
            arrowprops=dict(arrowstyle="-|>", color="#1565C0",
                            lw=1.2, linestyle="dashed"),
            zorder=4)
ax.text(0.62, 0.28, "V5.pt →\nDetection", fontsize=6.5, color="#1565C0",
        ha="center", style="italic")

# =====================================================================
#  6-class legend
# =====================================================================
cls_colors = ["#E57373", "#FFB74D", "#FFF176", "#81C784", "#64B5F6", "#BA68C8"]
cls_names  = ["Title", "Subtitle", "Text", "Image", "Table", "Equation"]
leg_x = 0.83
leg_y = 0.26
ax.text(leg_x, leg_y + 0.06, "6 Detection Classes", fontsize=8,
        weight="bold", color="#333")
for i, (cn, cc) in enumerate(zip(cls_names, cls_colors)):
    yy = leg_y + 0.04 - i * 0.028
    ax.add_patch(FancyBboxPatch((leg_x, yy - 0.008), 0.015, 0.018,
                                boxstyle="round,pad=0,rounding_size=0.004",
                                facecolor=cc, edgecolor="none", zorder=2))
    ax.text(leg_x + 0.02, yy + 0.001, cn, fontsize=7, va="center", color="#333")

# =====================================================================
#  Stage number badges
# =====================================================================
for i in range(7):
    cx = xs[i] + BOX_W / 2
    cy = ROW1_Y + BOX_H + 0.02
    circle = plt.Circle((cx, cy), 0.012, color="#333333", zorder=5)
    ax.add_patch(circle)
    ax.text(cx, cy, str(i + 1), ha="center", va="center",
            fontsize=7, color="white", weight="bold", zorder=6)

# =====================================================================
#  Save
# =====================================================================
out_path = "/Users/jinnwoook/Desktop/포토폴리오/pipeline_figure.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight",
            facecolor=C["bg"], edgecolor="none", pad_inches=0.15)
plt.close(fig)
print(f"Saved: {out_path}")
