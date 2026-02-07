# Method Pipeline: Visually-rich Document Understanding

> 논문 방법론(Methodology) 섹션의 파이프라인 그림 제작을 위한 정리 문서

---

## 1. Overall Pipeline Overview

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
│  Input      │───▶│  Document    │───▶│  Layout       │───▶│  Post-       │───▶│  OCR & Text  │───▶│  Reading    │───▶ Output
│  Document   │    │  Conversion  │    │  Detection    │    │  Processing  │    │  Extraction  │    │  Order      │    (CSV)
└─────────────┘    └──────────────┘    └───────────────┘    └──────────────┘    └──────────────┘    └─────────────┘
 PDF/PPTX/         Format-aware         Dual-model           NMS + Class        EasyOCR              Adaptive
 JPG/PNG           High-res (800DPI)    Strategy             Refinement         (ko+en)              Algorithm
```

---

## 2. Stage 1: Document Conversion (Input Processing)

### 목적
다양한 문서 포맷을 통일된 고해상도 이미지로 변환

### 흐름도

```
                    ┌──────────────────────────────────────┐
                    │         Input Document               │
                    │   (PDF / PPTX / JPG / PNG)           │
                    └──────────┬───────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
     ┌─────────────┐  ┌──────────────┐  ┌─────────────┐
     │    PDF       │  │    PPTX      │  │  JPG / PNG  │
     └──────┬──────┘  └──────┬───────┘  └──────┬──────┘
            │                │                  │
            ▼                ▼                  │
     ┌─────────────┐  ┌──────────────┐         │
     │  pdf2image   │  │  LibreOffice │         │
     │  @ 800 DPI   │  │  (headless)  │         │
     └──────┬──────┘  └──────┬───────┘         │
            │                ▼                  │
            │         ┌──────────────┐         │
            │         │  PDF → PNG   │         │
            │         │  @ 800 DPI   │         │
            │         └──────┬───────┘         │
            │                │                  │
            ▼                ▼                  ▼
     ┌──────────────────────────────────────────────┐
     │          PIL Image (per page)                 │
     │     + Document Metadata Extraction            │
     │     (is_pptx, is_landscape = W > H)           │
     └──────────────────────────────────────────────┘
```

### 핵심 파라미터
| 파라미터 | 값 | 설명 |
|---------|-----|------|
| DPI | 800 | 텍스트 품질 보존을 위한 고해상도 변환 |
| PPTX 중간 변환 | LibreOffice → PDF → PNG | 2단계 변환 |

---

## 3. Stage 2: Layout Detection (Dual-Model Strategy)

### 목적
문서 유형에 따라 최적화된 모델을 선택하여 레이아웃 요소 검출

### 흐름도

```
     ┌──────────────────────────────────────┐
     │         Document Type Check           │
     │   is_pptx OR is_landscape (W > H)?    │
     └─────────────┬────────────────────────┘
                   │
          ┌────────┴────────┐
          │ Yes             │ No
          ▼                 ▼
  ┌───────────────┐  ┌───────────────┐
  │  Fine-tuned   │  │  Base Model   │
  │  Model (V5)   │  │  (DocLayNet)  │
  │               │  │               │
  │  YOLOv12-L    │  │  YOLOv12-L    │
  │  Fine-tuned   │  │  Pre-trained  │
  │  on NeurIPS   │  │  on DocLayNet │
  │  Posters      │  │  (IBM)        │
  └───────┬───────┘  └───────┬───────┘
          │                   │
          ▼                   ▼
  ┌───────────────┐  ┌───────────────┐
  │ Confidence     │  │ Confidence     │
  │ Thresholds:    │  │ Thresholds:    │
  │ Title=0.45     │  │ Title=0.15     │
  │ Subtitle=0.10  │  │ Subtitle=0.15  │
  │ Text=0.10      │  │ Text=0.10      │
  │ Image=0.22     │  │ Struct=0.10    │
  │ Table=0.10     │  │                │
  │ Equation=0.10  │  │                │
  └───────┬───────┘  └───────┬───────┘
          │                   │
          └─────────┬─────────┘
                    ▼
     ┌──────────────────────────────┐
     │  Detection Results           │
     │  {bbox, class, confidence}   │
     │  per element                 │
     └──────────────────────────────┘
```

### Detection 세부 과정
```
Image (PIL) ──▶ Resize (1280×1280) ──▶ YOLO Inference (imgsz=1024) ──▶ Per-class Confidence Filter ──▶ Bbox Scale to Original
```

### 클래스 정의 (6 classes)
| Class | 설명 | DocLayNet 매핑 |
|-------|------|---------------|
| `title` | 문서 제목/메인 헤딩 | Title |
| `subtitle` | 섹션 헤더 | Section-header |
| `text` | 본문 텍스트/단락 | Text |
| `image` | 그림/사진 | Picture |
| `table` | 데이터 테이블 | Table |
| `equation` | 수학 공식 | Formula |

---

## 4. Stage 3: Post-Processing

### 목적
검출 결과의 중복 제거 및 분류 보정

### 흐름도

```
     ┌──────────────────────────┐
     │    Raw Detection Results  │
     └────────────┬─────────────┘
                  │
                  ▼
     ┌──────────────────────────┐
     │  (A) NMS                  │
     │  Non-Maximum Suppression  │
     │  IoU threshold = 0.5      │
     │  - Sort by confidence     │
     │  - Remove overlapping     │
     │    duplicate boxes        │
     └────────────┬─────────────┘
                  │
                  ▼
     ┌──────────────────────────┐
     │  (B) Classification       │
     │      Refinement           │
     │                           │
     │  [Base Model Only]        │
     │  Subtitle → Title         │
     │  Promotion                │
     │  (conf < 0.80 시 승격)    │
     └────────────┬─────────────┘
                  │
                  ▼
     ┌──────────────────────────┐
     │  (C) Enforce Single       │
     │      Top Title            │
     │                           │
     │  - 최상단·최좌측 title    │
     │    만 유지                 │
     │  - 나머지 title →         │
     │    subtitle로 변환        │
     └────────────┬─────────────┘
                  │
                  ▼
     ┌──────────────────────────┐
     │  (D) Caption Text         │
     │      Filtering            │
     │                           │
     │  패턴 매칭:               │
     │  "Figure X:", "Table X:", │
     │  "그림", "표"             │
     │  (conf < 0.95 시 제거)    │
     └────────────┬─────────────┘
                  │
                  ▼
     ┌──────────────────────────┐
     │  Refined Detection Items  │
     └──────────────────────────┘
```

---

## 5. Stage 4: OCR & Text Extraction

### 목적
검출된 텍스트 영역에서 한국어/영어 텍스트 추출 및 정제

### 흐름도

```
     ┌────────────────────────────────┐
     │   For each textual element     │
     │   (title, subtitle, text)      │
     └──────────────┬─────────────────┘
                    │
                    ▼
     ┌────────────────────────────────┐
     │  (A) Region Cropping           │
     │  bbox → crop from original     │
     │  high-res image                │
     └──────────────┬─────────────────┘
                    │
                    ▼
     ┌────────────────────────────────┐
     │  (B) EasyOCR Recognition       │
     │                                │
     │  ┌────────┐  ┌────────┐       │
     │  │ Korean │  │English │       │
     │  │ Model  │  │ Model  │       │
     │  │(ko_g2) │  │(en_g2) │       │
     │  └────┬───┘  └────┬───┘       │
     │       └──────┬─────┘           │
     │              ▼                 │
     │       Bilingual OCR            │
     │       (ko + en)                │
     └──────────────┬─────────────────┘
                    │
                    ▼
     ┌────────────────────────────────┐
     │  (C) Text Cleaning Pipeline    │
     │  (10-step)                     │
     │                                │
     │  1. NFKC Unicode 정규화        │
     │  2. 줄바꿈 통일 (\r\n → \n)   │
     │  3. Math/Script 태그 제거      │
     │  4. 하이픈-줄바꿈 병합         │
     │  5. HTML 태그 제거             │
     │  6. 불릿/넘버링 제거           │
     │     (•, ①, 1. 등 50+종)       │
     │  7. 하이픈 분리 단어 결합      │
     │  8. 줄바꿈 → 공백 변환        │
     │  9. 연속 공백 정규화           │
     │ 10. 양끝 공백 제거 (strip)     │
     └──────────────┬─────────────────┘
                    │
                    ▼
     ┌────────────────────────────────┐
     │  Clean Text per Element        │
     └────────────────────────────────┘
```

---

## 6. Stage 5: Reading Order Prediction (Adaptive Algorithm)

### 목적
문서 유형에 맞는 읽기 순서를 자동 결정

### 분기 흐름도

```
     ┌────────────────────────────────────────────┐
     │         Document Type Branch                │
     │   is_landscape OR is_pptx?                  │
     └─────────────────┬──────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │ Yes                     │ No
          ▼                         ▼
  ┌───────────────────┐    ┌───────────────────┐
  │  SCAN MODE        │    │  DEFAULT MODE     │
  │  (Landscape/PPTX) │    │  (Portrait/PDF)   │
  └───────────────────┘    └───────────────────┘
```

---

### 6-A. Default Mode (Portrait/PDF)

```
     ┌────────────────────────────────┐
     │  Document Layout               │
     │  Classification                │
     │                                │
     │  Text Span Detection:          │
     │  width ≥ 65% page_width?       │
     │                                │
     │  ≥3 spans → "general"          │
     │  <3 spans → "paper"            │
     └──────────┬─────────────────────┘
                │
       ┌────────┴────────┐
       ▼                  ▼
  ┌──────────┐     ┌──────────────┐
  │"general" │     │  "paper"     │
  │ layout   │     │  layout      │
  └────┬─────┘     └──────┬───────┘
       │                   │
       ▼                   ▼
  ┌──────────┐     ┌──────────────────────────────┐
  │ Simple   │     │ (1) Full-width Span 식별     │
  │ Top→     │     │     (텍스트 너비 ≥ 65%)      │
  │ Bottom,  │     │                               │
  │ Left→    │     │ (2) Span 기준 영역 분할       │
  │ Right    │     │     (Region Segmentation)      │
  │          │     │                               │
  │ Y좌표    │     │ (3) Per-region 다단 감지       │
  │ 기반     │     │     KMeans Clustering          │
  │ 그룹핑   │     │     (k=3 → k=2 → single)     │
  └────┬─────┘     │                               │
       │           │ (4) Column-wise 정렬           │
       │           │     Top→Bottom per column      │
       │           │     Left→Right across columns  │
       │           └──────────────┬────────────────┘
       │                          │
       └──────────┬───────────────┘
                  ▼
     ┌────────────────────────────────┐
     │     Ordered Element List       │
     └────────────────────────────────┘
```

### Multi-Column Detection 세부

```
  Items in Region
       │
       ▼
  Extract center_x
       │
       ▼
  ┌─────────────────────────┐
  │ Try KMeans k=3          │
  │ gap ≥ 22% page_width?   │──No──┐
  │ items_per_col ≥ 2?      │      │
  └───────┬─────────────────┘      │
          │ Yes                     ▼
          ▼                 ┌─────────────────────────┐
   3-Column Layout          │ Try KMeans k=2          │
                            │ gap ≥ 30% page_width?   │──No──▶ Single Column
                            │ items_per_col ≥ 2?      │
                            └───────┬─────────────────┘
                                    │ Yes
                                    ▼
                             2-Column Layout
```

---

### 6-B. Scan Mode (Landscape/PPTX)

```
     ┌────────────────────────────────────────────┐
     │  (1) Subtitle Row Grouping                  │
     │                                             │
     │  - Subtitle들의 y좌표 기반 행 그룹핑         │
     │  - Row tolerance = max(24px, median_h × 0.6)│
     │  - Vertical overlap ≥ 35% → 같은 행         │
     │  - 행 내부: left → right 정렬               │
     └─────────────────┬───────────────────────────┘
                       │
                       ▼
     ┌────────────────────────────────────────────┐
     │  (2) Top-to-Bottom Scan                     │
     │                                             │
     │  FOR EACH element (top → bottom):           │
     └─────────────────┬───────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
  ┌───────────────┐        ┌───────────────┐
  │  Subtitle?    │        │  Non-subtitle │
  │               │        │               │
  │  ▼            │        │  ▼            │
  │  Process      │        │  Process      │
  │  entire       │        │  region up    │
  │  subtitle     │        │  to next      │
  │  row          │        │  subtitle     │
  │               │        │  row          │
  └───────┬───────┘        └───────┬───────┘
          │                        │
          ▼                        ▼
  ┌──────────────────┐     ┌──────────────────┐
  │ Numeric Prefix   │     │ Unified Layout   │
  │ Detection        │     │ (Default Mode    │
  │                  │     │  algorithm 적용) │
  │ "1.", "2." 등    │     └──────────────────┘
  │ ≥2개 숫자 접두사  │
  │ → 숫자순 정렬    │
  │ 아니면 → L→R    │
  └───────┬──────────┘
          │
          ▼
  ┌──────────────────────────────────────┐
  │  Per-Subtitle Vertical Strip         │
  │                                      │
  │  ┌─────────┬─────────┬─────────┐    │
  │  │ Sub 1   │ Sub 2   │ Sub 3   │    │
  │  ├─────────┼─────────┼─────────┤    │
  │  │ Strip 1 │ Strip 2 │ Strip 3 │    │
  │  │ (items  │ (items  │ (items  │    │
  │  │  below  │  below  │  below  │    │
  │  │  sub 1) │  sub 2) │  sub 3) │    │
  │  │         │         │         │    │
  │  │ Paper-  │ Paper-  │ Paper-  │    │
  │  │ like    │ like    │ like    │    │
  │  │ reading │ reading │ reading │    │
  │  └─────────┴─────────┴─────────┘    │
  └──────────────────────────────────────┘
```

---

## 7. Stage 6: Output Generation

```
     ┌────────────────────────────────┐
     │  Ordered Elements              │
     │  + OCR Text                    │
     │  + Detection Metadata          │
     └──────────────┬─────────────────┘
                    │
                    ▼
     ┌────────────────────────────────┐
     │  CSV Output                    │
     │                                │
     │  Columns:                      │
     │  ┌───────────────────────────┐ │
     │  │ ID            │ doc_p1    │ │
     │  │ category_type │ title     │ │
     │  │ confidence    │ 0.95      │ │
     │  │ order         │ 0         │ │
     │  │ text          │ "Main.."  │ │
     │  │ bbox          │ x1,y1,..  │ │
     │  └───────────────────────────┘ │
     │                                │
     │  Encoding: UTF-8-sig           │
     └────────────────────────────────┘
```

---

## 8. Fine-tuning Pipeline (Training)

### 목적
Landscape/PPTX 문서에 최적화된 검출 모델 학습

```
  ┌──────────────────┐
  │ NeurIPS Poster   │
  │ Dataset          │
  │ (100 images)     │
  │ 6 classes        │
  │ YOLO format      │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ Train/Val Split  │
  │ 90% / 10%       │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐      ┌──────────────────┐
  │ Base Model       │      │ Training Config   │
  │ yolov12l-        │      │                   │
  │ doclaynet.pt     │      │ imgsz: 1024       │
  │ (Pre-trained)    │      │ batch: 16         │
  └────────┬─────────┘      │ epochs: 300       │
           │                 │ patience: 50      │
           ▼                 │ cos_lr: True      │
  ┌──────────────────┐      │                   │
  │ Fine-tuning      │◀─────│ Augmentation:     │
  │ (Transfer        │      │  mosaic=0.2       │
  │  Learning)       │      │  fliplr=0.0       │
  │                  │      │  scale=0.20       │
  │ Early Stop @     │      │  erasing=0.4      │
  │ Epoch 61         │      └──────────────────┘
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ V5.pt            │
  │ (Fine-tuned)     │
  │                   │
  │ mAP50: 0.852     │
  │ mAP50-95: 0.701  │
  └──────────────────┘
```

---

## 9. 논문 Figure 제작용 요약 (Simplified Pipeline)

논문에 삽입할 메인 파이프라인 그림은 아래 구조로 제작 권장:

```
┌─────────┐   ┌──────────┐   ┌─────────────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐   ┌────────┐
│  Input   │──▶│ Document │──▶│   Dual-Model    │──▶│  Post-   │──▶│ Bilingual│──▶│ Adaptive│──▶│ Output │
│ Document │   │Conversion│   │   Layout        │   │Processing│   │   OCR    │   │ Reading │   │  CSV   │
│          │   │          │   │   Detection     │   │          │   │          │   │  Order  │   │        │
│ PDF      │   │ pdf2image│   │                 │   │ NMS      │   │ EasyOCR  │   │         │   │ ID     │
│ PPTX     │   │ @ 800DPI │   │ ┌─────────────┐│   │ Class    │   │ Korean + │   │ Default │   │ class  │
│ JPG      │   │          │   │ │Portrait/PDF ││   │ Refine   │   │ English  │   │  Mode   │   │ conf   │
│ PNG      │   │LibreOffice│  │ │→ DocLayNet  ││   │ Caption  │   │          │   │  (PDF)  │   │ order  │
│          │   │ (PPTX)   │   │ ├─────────────┤│   │ Filter   │   │ 10-step  │   │         │   │ text   │
│          │   │          │   │ │Landscape/PPT││   │          │   │ Text     │   │ Scan    │   │ bbox   │
│          │   │          │   │ │→ Fine-tuned ││   │          │   │ Cleaning │   │  Mode   │   │        │
│          │   │          │   │ │  (V5.pt)    ││   │          │   │          │   │  (PPTX) │   │        │
│          │   │          │   │ └─────────────┘│   │          │   │          │   │         │   │        │
└─────────┘   └──────────┘   └─────────────────┘   └──────────┘   └──────────┘   └─────────┘   └────────┘
                                  YOLOv12-L                                        KMeans
                                  6 Classes                                        Multi-col
```

---

## 10. 핵심 기술적 기여 (Key Technical Contributions)

논문에서 강조할 주요 기여점:

| # | 기여 | 설명 |
|---|------|------|
| 1 | **Dual-Model Strategy** | 문서 유형(portrait/landscape)에 따른 적응적 모델 선택 |
| 2 | **Adaptive Reading Order** | Default Mode(multi-column detection) + Scan Mode(subtitle row grouping) |
| 3 | **Multi-Column Detection** | KMeans clustering 기반 자동 다단 감지 (2단/3단) |
| 4 | **Subtitle-driven Scan** | Landscape 문서에서 subtitle 기반 vertical strip 분할 + 숫자 접두사 정렬 |
| 5 | **Domain-specific Fine-tuning** | NeurIPS 포스터 100장으로 DocLayNet 모델 전이학습 (mAP50: 0.852) |
| 6 | **Bilingual OCR + 10-step Cleaning** | 한/영 이중 언어 OCR + 체계적 텍스트 후처리 파이프라인 |

---

## 11. 모델 및 하이퍼파라미터 요약

### Detection Models

| Model | Architecture | Training Data | Usage |
|-------|-------------|---------------|-------|
| `yolov12l-doclaynet.pt` | YOLOv12-L | DocLayNet (IBM) | Portrait/PDF 문서 |
| `V5.pt` | YOLOv12-L (fine-tuned) | NeurIPS Posters (100장) | Landscape/PPTX 문서 |

### Key Hyperparameters

| Category | Parameter | Value |
|----------|-----------|-------|
| **Conversion** | DPI | 800 |
| **Detection** | Input size | 1024 × 1024 |
| **NMS** | IoU threshold | 0.5 |
| **Multi-col** | Max columns | 3 |
| **Multi-col** | Gap ratio (2-col) | 30% |
| **Multi-col** | Gap ratio (3-col) | 22% |
| **Scan Mode** | Row overlap threshold | 35% |
| **Training** | Epochs (early stop) | 61 / 300 |
| **Training** | Batch size | 16 |
| **Training** | Patience | 50 |
