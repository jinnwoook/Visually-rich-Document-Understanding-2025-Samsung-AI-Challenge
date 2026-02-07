# -*- coding: utf-8 -*-
# Ultralytics YOLO 🚀, AGPL-3.0 license
## Ultralytics YOLO 🚀, AGPL-3.0 license
# https://huggingface.co/hantian/yolo-doclaynet/blob/main/yolov12l-doclaynet.pt
#### CRAFT 텍스트 검출 모델 (craft_mlt_25k.pth)

# - 용도: 텍스트 영역 검출
# - 라이선스: Apache-2.0
# - 라이선스 링크: https://github.com/clovaai/CRAFT-pytorch/blob/master/LICENSE
# - 다운로드: https://github.com/clovaai/CRAFT-pytorch

# ### 영어 텍스트 인식 모델 (english_g2.pth)

# - 용도: 영어 텍스트 인식
# - 라이선스: EasyOCR Apache-2.0
# - 라이선스 링크: https://github.com/JaidedAI/EasyOCR/blob/master/LICENSE
# - 다운로드 출처:
#     - EasyOCR GitHub Releases: https://github.com/JaidedAI/EasyOCR/releases

# ### 한국어 텍스트 인식 모델 (korean_g2.pth)

# - 용도: 한국어 텍스트 인식
# - 라이선스: EasyOCR Apache-2.0
# - 라이선스 링크: https://github.com/JaidedAI/EasyOCR/blob/master/LICENSE
# - 다운로드 출처:
#     - EasyOCR GitHub Releases: https://github.com/JaidedAI/EasyOCR/releases
# ============================================================
# YOLO (doclaynet 단일 전역 감지) + (PPTX/landscape는 finetune로 전역 감지) + EasyOCR
# - 좌표계: CSV의 (width,height) = "타깃 좌표"를 절대 기준으로 유지 (제출용)
# - 전역 NMS → 후처리(Subtitle→Title 승격, 단일 Top-Title) → 최종 OCR(EasyOCR)
# - PPTX 또는 가로형(landscape) 페이지는 yoloy12l_finetune.pt 로 "전역 감지" 수행
# - ROI 2차 감지 및 subtitle 기반 로컬 정렬 로직 제거
# - ⚠️ Surya 제거, OCR은 EasyOCR만 사용
# - ✅ NEW(scan-mode): 가로형에서 y↑ 스캔하면서
#     · subtitle 행(row)을 만나면 그 행의 subtitle들을 **숫자접두 2개↑면 숫자순**, 아니면 **좌→우**로 처리
#     · 각 subtitle은 자기 세로 스트립(좌/우 경계 + 다음 행 시작 전까지)을 **논문 읽기(멀티컬럼 감지 후 컬럼별 위→아래)**로 읽음
#     · subtitle 구간 외의 비-subtitle 구간은 기본 레이아웃 로직으로 읽음
# - 세로형은 기존 classify_doc_type → 기본 레이아웃(_order_default)
# ============================================================

import os, glob, re, subprocess, unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path
from ultralytics import YOLO
from sklearn.cluster import KMeans
import easyocr
import regex as rxx

# ---------------- Device ----------------
try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
    _DEVICE = "cuda:0" if _HAS_CUDA else "cpu"
except Exception:
    _HAS_CUDA = False
    _DEVICE = "cpu"

def _to_device_yolo(m):
    try: m.to(_DEVICE)
    except Exception: pass
    return m

# ---------------- Project Root ----------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)

# ---------------- Hyperparams ----------------
MODEL_DOCLAYNET_PATH = _PROJECT_ROOT + "/models/pretrained/yolov12l-doclaynet.pt"
MODEL_FINETUNE_PATH  = _PROJECT_ROOT + "/models/finetuned/V5.pt"    # forcing_best

CONF_TITLE    = 0.15
CONF_SUBTITLE = 0.15
CONF_TEXT     = 0.10
CONF_STRUCT   = 0.10

ROI_CONF_TITLE    = 0.45
ROI_CONF_SUBTITLE = 0.10
ROI_CONF_TEXT     = 0.10
ROI_CONF_TABLE    = 0.10
ROI_CONF_EQUATION = 0.10
ROI_CONF_IMAGE    = 0.22459314867074343

CONF_SUBTITLE_PROMOTE_TO_TITLE = 0.80
CAPTION_TEXT_FILTER_MIN_CONF   = 0.95

SAVE_VIS = False
VIS_DIR  = _PROJECT_ROOT + "/outputs/vis"

SPAN_WIDTH_RATIO = 0.65
SPAN_EXCLUDE_TOP_RATIO = 0.00
SPAN_EXCLUDE_BOTTOM_RATIO = 0.05
SPAN_GENERAL_THRESHOLD = 3

Y_TOL_RATIO = 0.15
MIN_LINE_TOL_PX = 8

MAX_COLS = 3
COL_GAP_RATIO = {2: 0.30, 3: 0.22}
MIN_ITEMS_PER_COL = {2: 2, 3: 2}

# ---- (선언만 유지: 현재 호출하지 않음) False table 관련 파라미터 ----
TABLE_MAX_AREA_RATIO   = 0.35
TABLE_MAX_WIDTH_RATIO  = 0.95
TABLE_MAX_HEIGHT_RATIO = 0.85
TABLE_TEXT_OVERLAP_AREA_RATIO = 0.55
TABLE_MIN_INNER_TEXT_BOXES    = 4
TABLE_LINE_IS_LONG_RATIO      = 0.70
TABLE_LONG_LINE_FRACTION_THR  = 0.60
FALSE_TABLE_ACTION = 'drop'

LABEL_MAP = {
    'Text': 'text', 'Title': 'title', 'Section-header': 'subtitle',
    'Formula': 'equation', 'Table': 'table', 'Picture': 'image',
    'text': 'text', 'title': 'title', 'section-header': 'subtitle',
    'subtitle': 'subtitle', 'formula': 'equation', 'equation': 'equation',
    'table': 'table', 'picture': 'image', 'image': 'image',
    'math': 'equation', 'mathematics': 'equation',
}

TITLE_SET     = {'title'}
SUBTITLE_SET  = {'subtitle'}
TEXT_SET      = {'text'}
TEXTUAL_SET   = {'title','subtitle','text'}
STRUCT_SET    = {'image', 'table', 'equation'}
FORM_SET      = {'equation'}

# ---------------- OCR (EasyOCR 전용) ----------------
try:
    reader = easyocr.Reader(
        ['ko','en'],
        gpu=_HAS_CUDA,
        model_storage_directory=_PROJECT_ROOT + '/models/pretrained',
        user_network_directory=_PROJECT_ROOT + '/models/pretrained',
        download_enabled=False,
        verbose=False
    )
except Exception:
    # GPU 초기화 실패 등 예외 발생 시 CPU로 폴백
    reader = easyocr.Reader(
        ['ko','en'],
        gpu=False,
        model_storage_directory=_PROJECT_ROOT + '/models/pretrained',
        user_network_directory=_PROJECT_ROOT + '/models/pretrained',
        download_enabled=False,
        verbose=False
    )

# ---------------- Models (중복 로딩 제거) ----------------
def _safe_load_model(*cands):
    for p in cands:
        if p and os.path.exists(p):
            try: return YOLO(p)
            except Exception: pass
    raise FileNotFoundError(f"모델 로딩 실패: {cands}")

model_doc       = _to_device_yolo(_safe_load_model(MODEL_DOCLAYNET_PATH, "./model/best.pt"))
model_finetune  = _to_device_yolo(_safe_load_model(MODEL_FINETUNE_PATH, "./model/yoloy12l_finetune.pt"))

# ---------------- I/O ----------------
def convert_to_images(input_path, temp_dir, dpi=800):
    ext = Path(input_path).suffix.lower()
    os.makedirs(temp_dir, exist_ok=True)
    if ext == ".pdf":
        return convert_from_path(input_path, dpi=dpi, output_folder=temp_dir, fmt="png")
    elif ext == ".pptx":
        subprocess.run(
            ["soffice", "--headless", "--convert-to", "pdf", "--outdir", temp_dir, input_path],
            check=True
        )
        pdf_path = os.path.join(temp_dir, Path(input_path).with_suffix(".pdf").name)
        return convert_from_path(pdf_path, dpi=dpi, output_folder=temp_dir, fmt="png")
    elif ext in [".jpg", ".jpeg", ".png"]:
        return [Image.open(input_path).convert("RGB")]
    else:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}")

# ---------------- Geometry ----------------
def scale_bbox_to_target(bbox, current_size, target_size):
    x1, y1, x2, y2 = bbox
    sx = target_size[0] / max(1, current_size[0])
    sy = target_size[1] / max(1, current_size[1])
    return [int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)]

def compute_iou(A,B):
    xA=max(A[0],B[0]); yA=max(A[1],B[1])
    xB=min(A[2],B[2]); yB=min(A[3],B[3])
    inter=max(0,xB-xA)*max(0,yB-yA)
    if inter==0: return 0.0
    aA=max(0,(A[2]-A[0]))*max(0,(A[3]-A[1]))
    aB=max(0,(B[2]-B[0]))*max(0,(B[3]-B[1]))
    den=aA+aB-inter
    return inter/den if den>0 else 0.0

def _area(b): return max(0,b[2]-b[0])*max(0,b[3]-b[1])
def _center_x(b): return 0.5*(b[0]+b[2])
def _center_y(b): return 0.5*(b[1]+b[3])

# ---------------- OCR utils & cleaning ----------------
_HTML_TAG_RE = rxx.compile(r"</?(?:b|strong|i|em|u|sub|sup|span|font|br|math|script|[A-Za-z0-9:_-]+)[^>]*>", rxx.IGNORECASE)
_WS_RE       = rxx.compile(r"\s+")
_HYPHEN_LINEBREAK_RE = rxx.compile(r"-\s*\n\s*")
_HYPHEN_SPLIT_RE     = rxx.compile(r"(\w)-\s+(\w)")
_MATH_TAG_RE         = rxx.compile(r"</?MATH[^>]*>|</?SCRIPT-[^>]*>", rxx.IGNORECASE)

_ICON_BULLETS = "•‣⁃–—•·‧∙⦁●○◌◍◎◉◯◈◆◇◊▣■□▪▫▮▯◼◻◽◾⬤⦿⦾▸▹►▻▾▿◁▷◀▶➔➜➝➞➟➠➡➤➥➧➨➩➪➯➲➳➵➸➼➽→⇒↦↪↳↠↣✓✔☑✗✘❌✚✛✜✢✣✤✦✧★☆✪✫✬✭✮✯✱✳✴❖❋❉❈❆❄⚫⚪"
_BULLET_LEAD_RE = rxx.compile(
    rf"(?m)^\s*(?:[{_ICON_BULLETS}\*\-\+]+|[A-Za-z][\.\)]|\d+[\.\)]|[①-⑳]|[❶-❾]|[➊-➓])\s+"
)
_BULLET_INLINE_RE = rxx.compile(rf"[{_ICON_BULLETS}]+")

def _clean_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _MATH_TAG_RE.sub("", s)
    s = _HYPHEN_LINEBREAK_RE.sub("", s)
    s = _HTML_TAG_RE.sub("", s)
    s = _BULLET_LEAD_RE.sub("", s)     # 숫자/불릿 제거(숫자판정은 raw로!)
    s = _BULLET_INLINE_RE.sub("", s)
    s = _HYPHEN_SPLIT_RE.sub(r"\1\2", s)
    s = s.replace("\n", " ")
    s = _WS_RE.sub(" ", s)
    return s.strip()

def extract_text_easy_raw(image_pil, bbox_target, target_size):
    x1,y1,x2,y2 = map(int, scale_bbox_to_target(bbox_target, target_size, image_pil.size))
    crop = image_pil.crop((x1,y1,x2,y2))
    arr = np.array(crop)
    try:
        res = reader.readtext(arr, detail=0)
        return (" ".join(res)).strip()
    except Exception:
        return ""

def extract_text_easy(image_pil, bbox_target, target_size):
    return _clean_text(extract_text_easy_raw(image_pil, bbox_target, target_size))

# ---------------- Caption filter ----------------
def _normalize_caption_text(s: str) -> str:
    t = unicodedata.normalize("NFKC", s or "")
    t = t.replace("–","-").replace("—","-").replace("−","-").replace("·",".")
    return t.strip()

_CAPTION_PATTERNS = [
    re.compile(r'^\s*(figure|fig\.)\s*([A-Za-z]{0,2}\s*\d+[A-Za-z]?|[IVXLCDM]+)\s*[:.\-\)\]]', re.IGNORECASE),
    re.compile(r'^\s*(table|tab\.)\s*([A-Za-z]{0,2}\s*\d+[A-Za-z]?|[IVXLCDM]+)\s*[:.\-\)\]]', re.IGNORECASE),
    re.compile(r'^\s*(그림)\s*\d+[가-힣A-Za-z]?\s*[:.\-\)\]]'),
    re.compile(r'^\s*(표)\s*\d+[가-힣A-Za-z]?\s*[:.\-\)\]]'),
]
_CAPTION_KEYWORDS = ["figure","fig.","table","tab.","그림","표"]

def _looks_like_caption_text(text: str) -> bool:
    if not text: return False
    t = _normalize_caption_text(text)
    for p in _CAPTION_PATTERNS:
        if p.search(t): return True
    prefix = t.split()[0] if t.split() else ""
    if any(prefix.startswith(k) for k in _CAPTION_KEYWORDS): return True
    return False

def filter_out_caption_like_text(items):
    kept, removed = [], 0
    for it in items:
        if it['category_type'] in TEXTUAL_SET:
            if it.get('confidence_score',0.0) <= CAPTION_TEXT_FILTER_MIN_CONF:
                if _looks_like_caption_text(it.get('text','')):
                    removed += 1; continue
        kept.append(it)
    if removed:
        print(f"[filter] caption-like textual boxes removed: {removed} (min_conf={CAPTION_TEXT_FILTER_MIN_CONF})")
    return kept

# ---------------- NMS ----------------
def remove_duplicate_boxes(pred_items, iou_thresh=0.5):
    pred_items = sorted(pred_items, key=lambda d: d['confidence_score'], reverse=True)
    kept=[]
    for it in pred_items:
        if all(compute_iou(it['bbox'], k['bbox']) <= iou_thresh for k in kept):
            kept.append(it)
    return kept

# ---------------- Doc type & base/global order ----------------
def classify_doc_type(pred_items, page_size):
    W,H = page_size
    def is_text_span(it):
        if it['category_type']!='text': return False
        x1,y1,x2,y2=it['bbox']; w=x2-x1
        return (w/max(1,W)) >= SPAN_WIDTH_RATIO
    top_cut=int(H*SPAN_EXCLUDE_TOP_RATIO)
    bot_cut=int(H*(1.0-SPAN_EXCLUDE_BOTTOM_RATIO))
    spans_mid=[it for it in pred_items if is_text_span(it) and it['bbox'][1]>top_cut and it['bbox'][3]<bot_cut]
    return "general" if len(spans_mid)>=SPAN_GENERAL_THRESHOLD else "paper"

def _to_lines_y1(sorted_items, line_tol):
    lines=[]
    for it in sorted_items:
        x1,y1,x2,y2=it['bbox']; placed=False
        for ln in lines:
            if abs(ln['y_top']-y1) <= line_tol:
                ln['items'].append(it); ln['ys'].append(y1)
                ln['y_top']=int(np.median(ln['ys'])); placed=True; break
        if not placed:
            lines.append({'y_top':y1,'ys':[y1],'items':[it]})
    for ln in lines:
        ln['items'].sort(key=lambda d:(d['bbox'][0],d['bbox'][1]))
    lines.sort(key=lambda L:L['y_top'])
    flat=[]
    for ln in lines: flat.extend(ln['items'])
    return flat

def _slice_by_yregion(items, y0, y1):
    out=[]
    for it in items:
        b=it['bbox']; cy=0.5*(b[1]+b[3])
        if y0<=cy<y1: out.append(it)
    return out

def _maybe_multi_cols(items, page_w, max_cols=MAX_COLS):
    n = len(items)
    if n < 2:
        return None
    xs = np.array([0.5 * (it['bbox'][0] + it['bbox'][2]) for it in items]).reshape(-1, 1)
    def _cluster_by_k(k):
        try:
            km = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(xs)
        except TypeError:
            km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(xs)
        centers = km.cluster_centers_.flatten()
        order = np.argsort(centers)
        centers_sorted = centers[order]
        relabel = {old: new for new, old in enumerate(order)}
        labels = np.array([relabel[l] for l in km.labels_])
        cols = [[] for _ in range(k)]
        for it, lb in zip(items, labels):
            cols[lb].append(it)
        return centers_sorted, cols
    k_max = min(max_cols, 3)
    for k in range(k_max, 1, -1):
        min_per_col = MIN_ITEMS_PER_COL.get(k, 2)
        if n < k * min_per_col: continue
        centers_sorted, cols = _cluster_by_k(k)
        gaps = np.diff(centers_sorted)
        gap_thr = COL_GAP_RATIO.get(k, 0.30) * page_w
        if all(g >= gap_thr for g in gaps) and all(len(c) >= min_per_col for c in cols):
            return cols
    return None

def _order_default(items, page_w, page_h, doc_type):
    if not items: return []
    heights=[abs(it['bbox'][3]-it['bbox'][1]) for it in items]
    avg_h=np.median(heights) if heights else 0
    line_tol=max(MIN_LINE_TOL_PX, int(avg_h*Y_TOL_RATIO))
    items_by_yx=sorted(items, key=lambda d:(d['bbox'][1],d['bbox'][0]))
    def is_span(it):
        x1,y1,x2,y2=it['bbox']; w=x2-x1
        return (w/max(1,page_w)) >= SPAN_WIDTH_RATIO
    if doc_type!="paper":
        ordered=_to_lines_y1(items_by_yx, line_tol)
    else:
        spans=[it for it in items_by_yx if is_span(it)]
        spans.sort(key=lambda d:d['bbox'][1])
        regions=[]; y_cursor=0
        for sp in spans:
            y_top,y_bot=sp['bbox'][1], sp['bbox'][3]
            if y_cursor<y_top: regions.append(("text",(y_cursor,y_top),None))
            regions.append(("span",(y_top,y_bot),sp)); y_cursor=y_bot
        if y_cursor<page_h: regions.append(("text",(y_cursor,page_h),None))
        ordered=[]
        def process_text_region(y0,y1):
            region_items=_slice_by_yregion([it for it in items_by_yx if not is_span(it)], y0,y1)
            if not region_items: return []
            multi_cols=_maybe_multi_cols(region_items, page_w, max_cols=MAX_COLS)
            if multi_cols is None:
                region_sorted=sorted(region_items, key=lambda d:(d['bbox'][1],d['bbox'][0]))
                return _to_lines_y1(region_sorted, line_tol)
            out=[]
            for col_items in multi_cols:
                col_sorted=sorted(col_items, key=lambda d:(d['bbox'][1],d['bbox'][0]))
                out.extend(_to_lines_y1(col_sorted, line_tol))
            return out
        for kind,(y0,y1),payload in regions:
            if kind=="text": ordered.extend(process_text_region(y0,y1))
            else: ordered.append(payload)
    return ordered



def assign_order_with_layout_unified(items, page_w, page_h):
    """
    '책/신문' 통합 읽기:
      - 가로로 긴 스팬(Text, 폭 비율>=SPAN_WIDTH_RATIO)이 있으면 그걸 경계로 영역 분할
      - 각 텍스트 영역은 멀티컬럼 감지 후 컬럼별 위→아래로 정렬
    """
    if not items:
        return []

    heights = [abs(it['bbox'][3]-it['bbox'][1]) for it in items]
    avg_h   = np.median(heights) if heights else 0
    line_tol = max(MIN_LINE_TOL_PX, int(avg_h*Y_TOL_RATIO))
    items_by_yx = sorted(items, key=lambda d:(d['bbox'][1], d['bbox'][0]))

    def is_span(it):
        if it['category_type'] != 'text':
            return False
        x1,y1,x2,y2 = it['bbox']
        w = x2 - x1
        return (w / max(1, page_w)) >= SPAN_WIDTH_RATIO

    # 스팬을 위→아래로 정렬해 영역 분할
    spans = [it for it in items_by_yx if is_span(it)]
    spans.sort(key=lambda d: d['bbox'][1])

    regions = []
    y_cursor = 0
    for sp in spans:
        y_top, y_bot = sp['bbox'][1], sp['bbox'][3]
        if y_cursor < y_top:
            regions.append(("text", (y_cursor, y_top)))
        regions.append(("span", (y_top, y_bot), sp))
        y_cursor = y_bot
    if y_cursor < page_h:
        regions.append(("text", (y_cursor, page_h)))
    if not regions:
        regions = [("text", (0, page_h))]

    ordered = []
    non_span = [it for it in items_by_yx if not is_span(it)]

    def process_text_region(y0, y1):
        region_items = _slice_by_yregion(non_span, y0, y1)
        if not region_items:
            return []
        multi_cols = _maybe_multi_cols(region_items, page_w, max_cols=MAX_COLS)
        if multi_cols is None:
            region_sorted = sorted(region_items, key=lambda d:(d['bbox'][1], d['bbox'][0]))
            return _to_lines_y1(region_sorted, line_tol)
        out = []
        for col_items in multi_cols:
            col_sorted = sorted(col_items, key=lambda d:(d['bbox'][1], d['bbox'][0]))
            out.extend(_to_lines_y1(col_sorted, line_tol))
        return out

    for reg in regions:
        if reg[0] == "text":
            y0, y1 = reg[1]
            ordered.extend(process_text_region(y0, y1))
        else:
            # span 자체도 순서에 포함 (원 코드 스타일 유지)
            ordered.append(reg[2])

    return ordered



# -------- NEW: region을 “논문 읽기” 방식으로 정렬(멀티컬럼 감지 → 컬럼별 위→아래) --------
def _order_region_paper_like(region_items, page_w, page_h, line_tol):
    if not region_items: return []
    multi_cols=_maybe_multi_cols(region_items, page_w, max_cols=MAX_COLS)
    if multi_cols is None:
        region_sorted=sorted(region_items, key=lambda d:(d['bbox'][1],d['bbox'][0]))
        return _to_lines_y1(region_sorted, line_tol)
    out=[]
    for col_items in multi_cols:
        col_sorted=sorted(col_items, key=lambda d:(d['bbox'][1],d['bbox'][0]))
        out.extend(_to_lines_y1(col_sorted, line_tol))
    return out

# ---------------- NEW: y↑ 스캔 기반 분기 로직 ----------------
ROW_Y_TOL_FACTOR = 0.6
MIN_ROW_TOL_PX   = 24
ROW_OVERLAP_THR  = 0.35

def _v_overlap_ratio(b1, b2):
    y1a, y2a = b1[1], b1[3]
    y1b, y2b = b2[1], b2[3]
    h1 = max(1, y2a - y1a); h2 = max(1, y2b - y1b)
    ov = max(0, min(y2a, y2b) - max(y1a, y1b))
    return ov / float(min(h1, h2))

def _group_subtitle_rows(subtitles, page_h):
    if not subtitles: return []
    h_list = [max(1, it['bbox'][3]-it['bbox'][1]) for it in subtitles]
    med_h  = np.median(h_list) if h_list else 20
    row_tol = max(MIN_ROW_TOL_PX, int(med_h * ROW_Y_TOL_FACTOR))
    rows = []
    for s in sorted(subtitles, key=lambda z: (z['bbox'][1], z['bbox'][0])):
        b = s['bbox']
        y_top, y_bot = b[1], b[3]
        y_cent = (y_top + y_bot) * 0.5
        placed = False
        for r in rows:
            band = [0, r['y0'], 0, r['y1']]
            if (_v_overlap_ratio(b, band) >= ROW_OVERLAP_THR) or (abs(y_cent - r['y_ref']) <= row_tol):
                r['subs'].append(s)
                r['y0'] = min(r['y0'], y_top)
                r['y1'] = max(r['y1'], y_bot)
                r['y_ref'] = int(0.5 * (r['y0'] + r['y1']))
                placed = True
                break
        if not placed:
            rows.append({'y0': y_top, 'y1': y_bot, 'y_ref': int(0.5 * (y_top + y_bot)), 'subs': [s]})
    rows.sort(key=lambda r: r['y_ref'])
    for r in rows:
        subs_by_x = sorted(r['subs'], key=lambda s: _center_x(s['bbox']))
        r['subs_by_x'] = subs_by_x
        r['centers_by_x'] = [_center_x(s['bbox']) for s in subs_by_x]
        r['idx_by_uid'] = {s['uid']: i for i, s in enumerate(subs_by_x)}
    return rows

_SUBTITLE_NUM_RE = rxx.compile(r"^\s*(\d+)\s*[\.\)]\s+")
def _subtitle_num_from_item(it):
    raw = (it.get('text_raw') or it.get('text') or "").strip()
    m = _SUBTITLE_NUM_RE.match(raw)
    return int(m.group(1)) if m else None

def assign_order_with_scan_modes(items, page_w, page_h):
    if not items: return []

    heights=[abs(it['bbox'][3]-it['bbox'][1]) for it in items]
    avg_h=np.median(heights) if heights else 0
    line_tol=max(MIN_LINE_TOL_PX, int(avg_h*Y_TOL_RATIO))
    items_by_yx=sorted(items, key=lambda d:(d['bbox'][1], d['bbox'][0]))

    subtitles=[it for it in items_by_yx if it['category_type']=="subtitle"]
    non_sub  =[it for it in items_by_yx if it['category_type']!="subtitle"]
    rows = _group_subtitle_rows(subtitles, page_h)

    ordered=[]; seen=set()
    def _key(it):
        uid = it.get('uid')
        return ('u', uid) if uid is not None else ('b', *it['bbox'])
    def _add(it):
        k=_key(it)
        if k in seen: return False
        seen.add(k); ordered.append(it); return True
    def _find_row_idx_by_sub(sub):
        for idx, r in enumerate(rows):
            for s in r['subs']:
                if s is sub or s['bbox'] == sub['bbox']:
                    return idx
        return None
    def _next_row_y0_after(y):
        for r in rows:
            if r['y0'] > y:
                return r['y0']
        return page_h

    def _process_subtitle_strip_in_row(ridx, sub):
        r = rows[ridx]
        idx_in_x = r['idx_by_uid'][sub['uid']]
        centers  = r['centers_by_x']
        left_bound  = 0 if idx_in_x==0 else 0.5*(centers[idx_in_x-1]+centers[idx_in_x])
        right_bound = page_w if idx_in_x==len(centers)-1 else 0.5*(centers[idx_in_x]+centers[idx_in_x+1])

        x1,y1,x2,y2 = sub['bbox']
        y_low = y2
        y_high = rows[ridx+1]['y0'] if (ridx+1)<len(rows) else page_h

        _add(sub)

        def _in_strip(it):
            bx1,by1,bx2,by2 = it['bbox']
            cx, cy = 0.5*(bx1+bx2), 0.5*(by1+by2)
            return (cy >= y_low) and (cy < y_high) and (cx >= left_bound) and (cx < right_bound)

        region_items = [it for it in non_sub if _key(it) not in seen and _in_strip(it)]
        # ▶ 여기서 논문 읽기 방식으로!
        for it in _order_region_paper_like(region_items, page_w, page_h, line_tol):
            _add(it)

    def _process_entire_row_with_rule(ridx):
        r = rows[ridx]
        nums = []
        for s in r['subs']:
            n = _subtitle_num_from_item(s)
            if n is not None:
                nums.append((n, s))
        if len(nums) >= 2:
            subs_order = [s for _, s in sorted(nums, key=lambda p:p[0])]
            rest = [s for s in r['subs_by_x'] if s not in subs_order]
            subs_order.extend(rest)
        else:
            subs_order = r['subs_by_x']
        for sub in subs_order:
            if _key(sub) not in seen:
                _process_subtitle_strip_in_row(ridx, sub)

    def _process_non_sub_block(start_item):
        bx1,by1,bx2,by2 = start_item['bbox']
        y0 = 0.5*(by1+by2)
        y1 = _next_row_y0_after(y0)
        region = [it for it in items_by_yx
                  if _key(it) not in seen
                  and it['category_type'] != 'subtitle'
                  and (0.5*(it['bbox'][1]+it['bbox'][3]) >= y0)
                  and (0.5*(it['bbox'][1]+it['bbox'][3]) <  y1)]
        if not region:
            _add(start_item); return
        # 비-sub 구간은 기존(신문/책 통합)으로 유지
        for it in assign_order_with_layout_unified(region, page_w, page_h):
            _add(it)

    for it in items_by_yx:
        if _key(it) in seen: continue
        if it['category_type']=="subtitle":
            ridx = _find_row_idx_by_sub(it)
            if ridx is None:
                _add(it)
            else:
                _process_entire_row_with_rule(ridx)
        else:
            _process_non_sub_block(it)

    for i,it in enumerate(ordered):
        it['order']=i
    return ordered

# ---------------- Per-image inference ----------------
def detect_all_single(model, image_pil, target_size, per_class_conf_map):
    resized = image_pil.resize((1280,1280))
    res = model(source=np.array(resized), imgsz=1024,
                conf=min(per_class_conf_map.values()) if per_class_conf_map else 0.0,
                verbose=False)[0]
    out=[]
    for box,score,cls in zip(res.boxes.xyxy, res.boxes.conf, res.boxes.cls):
        raw=res.names[int(cls)]
        mapped=LABEL_MAP.get(raw)
        if mapped is None: continue
        if per_class_conf_map and (score < per_class_conf_map.get(mapped, 0.0)): continue
        x1,y1,x2,y2 = scale_bbox_to_target(box.tolist(), (1280,1280), target_size)
        out.append({
            'category_type': mapped,
            'confidence_score': float(score.cpu().item()) if hasattr(score,"cpu") else float(score),
            'bbox': [int(x1),int(y1),int(x2),int(y2)]
        })
    return out

def enforce_single_top_title(items):
    titles=[it for it in items if it['category_type']=='title']
    if len(titles)>=2:
        top=sorted(titles, key=lambda it:(it['bbox'][1], it['bbox'][0], -(it['bbox'][2]-it['bbox'][0])))[0]
        for it in titles:
            if it is not top: it['category_type']='subtitle'
    return items

def inference_one_image(id_val, image_pil, target_size, vis_dir=VIS_DIR, is_pptx=False, is_landscape=False):
    Wt,Ht = target_size

    use_finetune = (is_pptx or is_landscape)
    page_model = model_finetune if use_finetune else model_doc

    if use_finetune:
        per_class_thr = {
            'title': ROI_CONF_TITLE, 'subtitle': ROI_CONF_SUBTITLE, 'text': ROI_CONF_TEXT,
            'table': ROI_CONF_TABLE, 'equation': ROI_CONF_EQUATION, 'image': ROI_CONF_IMAGE,
        }
    else:
        per_class_thr = {
            'title': CONF_TITLE, 'subtitle': CONF_SUBTITLE, 'text': CONF_TEXT,
            'table': CONF_STRUCT, 'equation': CONF_STRUCT, 'image': CONF_STRUCT,
        }

    pred_items = detect_all_single(page_model, image_pil, (Wt,Ht), per_class_thr)
    pred_items = remove_duplicate_boxes(pred_items, iou_thresh=0.5)
    if not pred_items: return []

    if not use_finetune:
        for it in pred_items:
            if it['category_type'] == 'subtitle' and it['confidence_score'] < CONF_SUBTITLE_PROMOTE_TO_TITLE:
                it['category_type'] = 'title'
    pred_items = enforce_single_top_title(pred_items)

    for i,it in enumerate(pred_items):
        it['uid'] = f"it-{i}"
        if it['category_type'] in TEXTUAL_SET:
            it['text_raw'] = extract_text_easy_raw(image_pil, it['bbox'], (Wt,Ht))
            it['text']     = _clean_text(it['text_raw'])
        else:
            it['text_raw'] = ""
            it['text']     = ""

    pred_items = filter_out_caption_like_text(pred_items)
    pred_items = remove_duplicate_boxes(pred_items, iou_thresh=0.5)
    if not pred_items: return []

    doc_type = classify_doc_type(pred_items, (Wt,Ht))
    if is_landscape:
        final_order = assign_order_with_scan_modes(pred_items, Wt, Ht)
    else:
        final_order = _order_default(pred_items, Wt, Ht, doc_type)

    for it in final_order:
        if it['category_type'] in TEXTUAL_SET:
            it['text'] = extract_text_easy(image_pil, it['bbox'], (Wt,Ht))

    if SAVE_VIS:
        os.makedirs(VIS_DIR, exist_ok=True)
        draw_bounding_boxes(image_pil.copy(), final_order, os.path.join(VIS_DIR, f"{id_val}_det.png"), target_size=(Wt,Ht))

    predictions=[]
    for idx,it in enumerate(final_order):
        x1,y1,x2,y2 = map(int, it['bbox'])
        predictions.append({
            'ID': id_val,
            'category_type': it['category_type'],
            'confidence_score': it['confidence_score'],
            'order': idx,
            'text': it.get('text','') if it['category_type'] in TEXTUAL_SET else '',
            'bbox': f'{x1}, {y1}, {x2}, {y2}'
        })
    return predictions

# ---------------- Visualization ----------------
def draw_bounding_boxes(image_pil, items, save_path, target_size):
    cmap={'text':(66,135,245),'title':(155,89,182),'subtitle':(23,190,207),
          'table':(46,204,113),'image':(243,156,18),'equation':(231,76,60)}
    Wimg,Himg=image_pil.size
    thickness=max(3,int(min(Wimg,Himg)*0.004))
    try: font=ImageFont.truetype("arial.ttf", max(14,int(thickness*4)))
    except: font=ImageFont.load_default()
    base=image_pil.convert("RGBA")
    overlay=Image.new("RGBA", base.size, (0,0,0,0))
    draw=ImageDraw.Draw(overlay)
    for order,it in enumerate(items):
        b_t = it['bbox']
        x1,y1,x2,y2 = map(int, scale_bbox_to_target(b_t, target_size, image_pil.size))
        cat=it['category_type']; conf=it['confidence_score']
        color=cmap.get(cat,(255,0,0))
        for k in range(thickness):
            draw.rectangle([x1-k,y1-k,x2+k,y2+k], outline=(*color,255), width=1)
        label=f"{order} | {cat} ({conf:.2f})"
        x1b, y1b, x2b, y2b = draw.textbbox((0,0), label, font=font)
        tw, th = (x2b - x1b), (y2b - y1b)
        pad=max(4,thickness)
        bx1,by1=x1,max(0,y1-th-pad*2); bx2,by2=x1+tw+pad*2,by1+th+pad*2
        draw.rectangle([bx1,by1,bx2,by2], fill=(0,0,0,160))
        draw.text((bx1+pad,by1+pad), label, font=font, fill=(255,255,255,255))
    out=Image.alpha_composite(base,overlay).convert("RGB")
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    out.save(save_path, quality=95)

# ---------------- Pipeline ----------------
def convert_to_images_safe(file_path, temp_image_dir, dpi=800):
    return convert_to_images(file_path, temp_dir=temp_image_dir, dpi=dpi)

def inference(test_csv_path=_PROJECT_ROOT + "/data/test.csv", output_csv_path=_PROJECT_ROOT + "/outputs/submission.csv"):
    output_dir=os.path.dirname(output_csv_path)
    os.makedirs(output_dir, exist_ok=True)

    temp_image_dir=_PROJECT_ROOT + "/temp_images"
    os.makedirs(temp_image_dir, exist_ok=True)

    csv_dir=os.path.dirname(test_csv_path)
    test_df = pd.read_csv(test_csv_path, dtype={"path": str})
    all_preds=[]

    for _,row in test_df.iterrows():
        id_val=row['ID']
        raw_path=row['path']
        if pd.isna(raw_path):
            print(f"⚠️ ID={id_val}: path가 비어있어 스킵")
            continue
        file_path=os.path.normpath(os.path.join(csv_dir, str(raw_path)))
        try:
            target_width  = int(row['width'])
            target_height = int(row['height'])
        except Exception:
            print(f"⚠️ ID={id_val}: width/height 불량 → 스킵")
            continue

        if not os.path.exists(file_path):
            print(f"⚠️ 파일 없음: {file_path}")
            continue

        try:
            is_pptx = Path(file_path).suffix.lower()==".pptx"
            is_landscape = (target_width > target_height)
            images = convert_to_images_safe(file_path, temp_image_dir)
            for i,image in enumerate(images):
                full_id=f"{id_val}_p{i+1}" if len(images)>1 else id_val
                preds=inference_one_image(
                    full_id, image, (target_width,target_height),
                    vis_dir=VIS_DIR, is_pptx=is_pptx, is_landscape=is_landscape
                )
                all_preds.extend(preds)
            print(f"✅ 예측 완료: {file_path}")
        except Exception as e:
            print(f"❌ 처리 실패: {file_path} → {e}")

    result_df=pd.DataFrame(all_preds)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    result_df.to_csv(output_csv_path, index=False, encoding='UTF-8-sig')
    print(f"✅ 저장 완료: {output_csv_path}")

# ---------------- Main ----------------
if __name__=="__main__":
    inference(_PROJECT_ROOT + "/data/test.csv", _PROJECT_ROOT + "/outputs/submission.csv")
