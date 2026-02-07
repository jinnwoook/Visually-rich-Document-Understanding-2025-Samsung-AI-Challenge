# Visually-rich Document Understanding을 위한 멀티모달 딥러닝 기반 On-device 문서 이해 시스템

**2025 Samsung AI Challenge 참가 보고서**

---

## Abstract

Visually-rich Document Understanding은 단순한 텍스트 인식을 넘어, 문서 내 레이아웃 구조와 시각적 표현, 그리고 읽기 흐름을 종합적으로 이해해야 하는 고난도 문제이다. 본 연구에서는 2025 Samsung AI Challenge를 대상으로, Layout Detection, Reading Order Prediction, OCR(Text Extraction)을 통합 수행하는 On-device 문서 이해 시스템을 제안한다.

제안 시스템의 핵심은 다음과 같다. 첫째, 문서 방향(세로형/가로형)에 따라 사전학습 모델과 파인튜닝 모델을 자동 선택하는 **적응형 이중 모델 전략(Adaptive Dual-Model Strategy)**을 도입하였다. 둘째, 반복적 오탐지 패턴을 교정하는 **규칙 기반 후처리(Rule-based Post-processing)**를 설계하였다. 셋째, 인간의 독서 습관을 모방한 세 가지 **구조 기반 읽기 순서 알고리즘**(책 읽기, 신문 읽기, 포스터 읽기)을 제안하였다. 넷째, DPI 최적화와 경량 OCR 엔진 선택을 통해 **추론 속도와 정확도의 균형**을 달성하였다.

실험 결과, 제안 시스템은 Baseline 대비 Public Score **292% 향상**(0.1168 → 0.4577)을 달성하였으며, OCR 추론 속도에서 **2.7배 개선**을 기록하였다. 전체 모델 크기는 221MB로 On-device 배포가 가능한 수준이다.

---

## 1. Introduction

현대 문서는 텍스트뿐만 아니라 제목, 표, 수식, 이미지, 그리고 강조 표현 등 다양한 시각적 요소에 의해 구성된다. 인간은 이러한 요소들을 종합적으로 해석하여 문서의 의미와 의도를 이해하지만, 기존 OCR 중심의 문서 처리 기술은 주로 텍스트 추출에 머무르며 문서 구조와 강조 의도를 충분히 반영하지 못하는 한계를 가진다.

Samsung AI Challenge의 Visually-rich Document Understanding 트랙은 이러한 한계를 극복하기 위해 문서 내 **객체 탐지(Layout Detection)**, **텍스트 추출(OCR)**, **읽기 순서(Reading Order)**를 통합적으로 평가한다. 특히 Layout Detection 이후 OCR과 Reading Order가 순차적으로 수행되므로, 초기 레이아웃 탐지 성능이 전체 파이프라인 성능을 좌우하는 핵심 요소가 된다.

본 연구의 목표는 **정확도와 추론 속도를 동시에 고려한 On-device 환경용 문서 이해 시스템**을 구축하는 것이다. 이를 위해 문서 형태에 따른 적응형 모델 선택, 규칙 기반 후처리, 구조 중심의 읽기 순서 알고리즘, 그리고 경량 OCR 최적화를 결합한 종합 파이프라인을 설계하였다.

본 논문의 주요 기여는 다음과 같다:

1. **적응형 이중 모델 전략**: 세로형 문서(PDF, 논문)와 가로형 문서(PPTX, 포스터)의 구조적 차이를 인식하고, 각 유형에 최적화된 모델을 자동 선택하는 방식을 제안한다.
2. **구조 기반 읽기 순서 알고리즘**: 의미론적 분석 없이 레이아웃 구조만으로 인간의 독서 패턴을 모방하는 세 가지 읽기 전략(책·신문·포스터 읽기)을 설계한다.
3. **Optuna 기반 신뢰도 임계값 최적화**: 클래스별 confidence threshold를 대회 점수를 목적 함수로 자동 최적화하여 탐지 성능을 극대화한다.
4. **On-device 실행 가능한 경량 시스템**: 총 221MB 모델 크기와 EasyOCR 기반 경량 OCR로 실용적 배포가 가능한 시스템을 구현한다.

---

## 2. Methodology

### 2.1 System Overview

제안 시스템은 Figure 1과 같이 5단계의 순차 파이프라인으로 구성된다.

```
입력 문서 (PDF / PPTX / Image)
        │
        ▼
┌──────────────────────────────────────┐
│  Stage 1. Document Conversion        │
│  PDF → 800 DPI PNG (pdf2image)       │
│  PPTX → LibreOffice → PDF → PNG     │
└──────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│  Stage 2. Adaptive Model Selection   │
│  세로형 → YOLOv12l-DocLayNet (Base)  │
│  가로형 → YOLOv12l Fine-tuned (V5)   │
└──────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│  Stage 3. Layout Detection           │
│  Global Detection (1280×1280 resize) │
│  + Rule-based Post-processing        │
└──────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│  Stage 4. Reading Order Prediction   │
│  세로형: 책/신문 읽기 알고리즘        │
│  가로형: 포스터 읽기 알고리즘 (Scan)  │
└──────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│  Stage 5. OCR Text Extraction        │
│  EasyOCR (한국어 + 영어)              │
│  + Text Cleaning Pipeline            │
└──────────────────────────────────────┘
        │
        ▼
    최종 출력 (CSV)
```

> **Figure 1.** 제안 시스템의 전체 파이프라인 아키텍처. Layout Detection의 결과가 Reading Order와 OCR의 입력이 되므로, 상위 단계의 정확도가 전체 성능을 좌우한다.

### 2.2 Adaptive Dual-Model Strategy

세로형 문서(논문, 보고서 등)와 가로형 문서(프레젠테이션, 학술 포스터 등)는 레이아웃 구성이 근본적으로 상이하다. 세로형 문서는 단일 또는 이중 컬럼의 정형화된 구조를 가지는 반면, 가로형 문서는 자유로운 배치와 시각적 강조 요소가 혼재한다. 이러한 도메인 간극을 해소하기 위해, 본 연구에서는 문서 방향에 따라 모델을 자동 선택하는 이중 모델 전략을 도입하였다.

**모델 선택 기준:**

| 조건 | 선택 모델 | 설명 |
|------|-----------|------|
| `파일 형식 == PPTX` 또는 `width > height` | Fine-tuned Model (V5.pt) | 가로형 문서 특화 |
| 그 외 (PDF, 세로형 이미지) | Base Model (yolov12l-doclaynet.pt) | 범용 문서 레이아웃 |

**Base Model**은 IBM DocLayNet 데이터셋으로 사전학습된 YOLOv12-Large를 그대로 사용하며, 세로형 문서의 정형화된 레이아웃에 대해 우수한 성능을 보인다. **Fine-tuned Model**은 동일 아키텍처를 NeurIPS 2024 학술 포스터 100장으로 추가 학습하여 가로형 문서의 비정형 레이아웃에 적응시킨 모델이다.

**클래스별 신뢰도 임계값 분리:**

각 모델의 특성에 맞게 클래스별 confidence threshold를 독립적으로 설정하였다. 특히 Fine-tuned Model에서는 Title의 임계값을 0.45로 높여 오탐을 억제하고, Base Model에서는 0.15로 낮추어 재현율을 확보하였다.

| Category | Base Model | Fine-tuned Model |
|----------|-----------|------------------|
| Title | 0.15 | 0.45 |
| Subtitle | 0.15 | 0.10 |
| Text | 0.10 | 0.10 |
| Table | 0.10 | 0.10 |
| Equation | 0.10 | 0.10 |
| Image | 0.10 | 0.22 |

> **Table 1.** 모델별 클래스 신뢰도 임계값. Optuna를 활용하여 대회 Public Score를 목적 함수로 최적화하였다.

### 2.3 Fine-tuning Strategy

#### 2.3.1 Training Dataset

가로형 문서 전용 학습 데이터셋을 다음과 같이 구축하였다.

| 항목 | 내용 |
|------|------|
| 데이터 소스 | NeurIPS 2024 학술 포스터 |
| 구성 | 대회 제공 PPT + 자체 제작 포스터 21장 + 외부 수집 79장 |
| 총 이미지 수 | 100장 |
| 어노테이션 형식 | YOLO Detection Format (bbox + class) |
| 라이선스 | CC BY 4.0 |
| 데이터 분할 | Train 90장 (90%) / Validation 10장 (10%) |

> **Table 2.** Fine-tuning 데이터셋 구성. 6개 클래스(Title, Subtitle, Text, Table, Image, Equation)에 대해 어노테이션을 수행하였다.

#### 2.3.2 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base Model | YOLOv12-Large (DocLayNet pretrained) | 문서 도메인 사전지식 활용 |
| Input Size | 1024 × 1024 | 고해상도 요소 탐지 |
| Batch Size | 16 | GPU 메모리 최적 활용 |
| Max Epochs | 300 (Early Stop at 61) | Patience=50으로 과적합 방지 |
| Optimizer | SGD (Auto) | lr=0.01, momentum=0.937 |
| LR Scheduler | Cosine Annealing | lrf=0.01 |
| Warmup | 3 epochs | 안정적 학습 시작 |
| Weight Decay | 0.0005 | 정규화 |
| AMP | True | Mixed Precision Training |
| Seed | 42 | 재현성 보장 |

> **Table 3.** Fine-tuning 학습 하이퍼파라미터.

#### 2.3.3 Domain-specific Data Augmentation

문서 도메인의 특수성을 고려하여 **보수적 증강 전략**을 채택하였다. 핵심 원칙은 텍스트의 가독성과 문서의 구조적 정합성을 보존하는 것이다.

| Augmentation | Value | Enabled | Rationale |
|-------------|-------|---------|-----------|
| Mosaic | 0.2 | O (제한적) | 다양한 레이아웃 조합 학습, 마지막 10 epoch 비활성화 |
| Horizontal Flip | 0.0 | **X** | 텍스트 방향 및 읽기 순서 보존 |
| Vertical Flip | 0.0 | **X** | 문서 상하 구조 보존 |
| Rotation | 0° | **X** | 정렬된 문서 구조 유지 |
| Translation | ±5% | O | 미세 위치 변화에 대한 일반화 |
| Scale | ±20% | O | 다양한 해상도 대응 |
| Erasing (Cutout) | 0.4 | O | 부분 가림에 대한 강건성 확보 |
| HSV Saturation | 0.30 | O | 색상 변화에 대한 강건성 |
| HSV Value | 0.20 | O | 밝기 변화에 대한 강건성 |
| HSV Hue | 0.0 | **X** | 색상 의미 보존 |
| MixUp / CutMix | 0.0 | **X** | 문서 구조 왜곡 방지 |

> **Table 4.** 도메인 특화 데이터 증강 전략. 기하학적 변환(뒤집기, 회전)을 비활성화하여 문서의 구조적 정합성을 보존하였다.

### 2.4 Rule-based Post-processing

사전학습 및 파인튜닝 모델의 탐지 결과를 분석한 결과, 다음과 같은 **반복적 오류 패턴**이 관찰되었다:

1. **중복 탐지(Duplicate Detection)**: 동일 영역에 대한 복수 바운딩 박스 생성
2. **Title-Subtitle 오분류**: Title이 Subtitle로 잘못 분류되는 현상
3. **복수 Title 탐지**: 한 페이지에 여러 개의 Title이 검출되는 현상
4. **Caption 오탐지**: "Figure X:", "Table X:" 등 캡션이 Text로 탐지되는 현상

이에 따라 다음 네 가지 규칙 기반 후처리를 순차 적용하였다:

| Rule | Description | Parameter |
|------|-------------|-----------|
| **R1. NMS** | IoU 기반 중복 박스 제거 | IoU threshold = 0.5 |
| **R2. Subtitle→Title 승격** | 낮은 신뢰도의 Subtitle을 Title로 재분류 | conf < 0.80 시 승격 |
| **R3. Single Top-Title** | 페이지당 Title 1개 제한, 나머지는 Subtitle로 변경 | 최상단·최좌측 Title 유지 |
| **R4. Caption Filtering** | 캡션 패턴의 Text 박스 제거 | conf < 0.95 + 패턴 매칭 |

> **Table 5.** Rule-based Post-processing 규칙. 관찰된 오류 패턴에 대응하여 설계하였다.

Caption 필터링에는 다음 정규식 패턴을 사용하였다:
- `Figure/Fig. + 숫자 + 구분자`
- `Table/Tab. + 숫자 + 구분자`
- `그림/표 + 숫자 + 구분자` (한국어)

### 2.5 Reading Order Algorithm

본 연구에서는 의미론적 문맥 분석(semantic analysis) 대신, **인간의 실제 독서 습관을 모방한 구조 기반 읽기 순서 알고리즘**을 제안한다. 문서 유형에 따라 세 가지 전략을 계층적으로 적용한다.

#### 2.5.1 Strategy 1: Book-style Reading (책 읽기)

단일 컬럼 문서에 적용되는 기본 전략이다.

```
Algorithm 1: Book-style Reading
─────────────────────────────────
Input: detected items, page size
Output: ordered items

1. Sort all items by (y_top, x_left)
2. Group items into lines by y-tolerance
   - tolerance = max(8px, median_height × 0.15)
3. Within each line: sort left → right
4. Order lines: top → bottom
```

#### 2.5.2 Strategy 2: Newspaper-style Reading (신문 읽기)

멀티 컬럼 문서에 적용되는 전략이다. KMeans 클러스터링으로 컬럼을 자동 감지한다.

```
Algorithm 2: Newspaper-style Reading
─────────────────────────────────────
Input: detected items, page_width
Output: ordered items

1. Detect text spans (width ≥ 65% of page_width)
2. Use spans as region boundaries
3. For each region between spans:
   a. Cluster center-x positions via KMeans (k=2,3)
   b. Validate column gaps:
      - 2-col: gap ≥ 30% × page_width
      - 3-col: gap ≥ 22% × page_width
   c. If valid columns detected:
      Read column-by-column (left→right),
      within each column: top→bottom
   d. Else: fallback to Book-style
4. Insert spans in encountered order
```

**문서 유형 자동 판별:**
- 페이지 너비의 65% 이상인 텍스트 스팬이 3개 이상 → **"general"** (책 읽기)
- 3개 미만 → **"paper"** (신문 읽기, 멀티컬럼 감지 적용)

#### 2.5.3 Strategy 3: Poster-style Reading (포스터/창자 읽기) — Scan-Mode

가로형 문서(PPTX, 학술 포스터)에 적용되는 본 연구의 핵심 알고리즘이다. Subtitle을 영역 분할의 기준점으로 활용하여, 포스터의 섹션 구조를 자동으로 인식한다.

```
Algorithm 3: Poster-style Reading (Scan-Mode)
──────────────────────────────────────────────
Input: detected items, page size
Output: ordered items

Phase 1. Subtitle Row Grouping
  1. Group subtitles with similar y-positions into rows
     - Row tolerance: max(24px, median_height × 0.6)
     - Vertical overlap threshold: 0.35
  2. Sort rows top → bottom

Phase 2. Y-ascending Scan
  FOR EACH item (sorted by y-coordinate ascending):
    CASE subtitle:
      Process the entire subtitle row:
      a. If ≥2 numeric prefixes (e.g., "1.", "2."):
         → Sort by number (ascending)
      b. Else:
         → Sort left → right
      c. For each subtitle in order:
         - Define vertical strip:
           · left/right: midpoints to adjacent subtitles
           · top: subtitle bottom edge
           · bottom: next row's top edge
         - Read strip using Newspaper-style (multi-column)
    CASE non-subtitle:
      Collect items until next subtitle row
      → Apply unified layout logic (Book/Newspaper)
```

> **Figure 2.** Poster-style Reading의 핵심 아이디어. Subtitle 행을 기준으로 문서를 수평 영역으로 분할하고, 각 Subtitle 아래의 세로 스트립을 독립적으로 읽는다.

### 2.6 OCR Text Extraction

#### 2.6.1 OCR Engine Selection

OCR 엔진 선택에 있어 정확도와 추론 속도의 균형을 고려하여 **EasyOCR**을 채택하였다. Surya OCR과의 비교 실험 결과, EasyOCR이 유사한 정확도에서 약 2.7배 빠른 추론 속도를 보였다 (Section 5.4 참조).

| Component | Model | License |
|-----------|-------|---------|
| Text Detection | CRAFT (craft_mlt_25k.pth) | Apache-2.0 |
| English Recognition | english_g2.pth | Apache-2.0 |
| Korean Recognition | korean_g2.pth | Apache-2.0 |

> **Table 6.** OCR 파이프라인 구성 요소.

#### 2.6.2 DPI Optimization

OCR 성능은 입력 이미지의 해상도에 크게 의존한다. 실험적으로 DPI를 200에서 800으로 증가시켰을 때, 추가 학습 없이 문자 누락 감소 및 문장 단위 복원 정확도가 크게 개선됨을 확인하였다 (Section 5.4.2 참조).

#### 2.6.3 Text Cleaning Pipeline

OCR 추출 결과에 대해 10단계 텍스트 클리닝을 적용하였다:

| Step | Operation | Example |
|------|-----------|---------|
| 1 | NFKC Unicode Normalization | ﬁ → fi |
| 2 | Line-break Unification | \r\n → \n |
| 3 | Math/Script Tag Removal | \<MATH\>...\</MATH\> → (제거) |
| 4 | Hyphen-linebreak Merge | "hyphen-\nword" → "hyphenword" |
| 5 | HTML Tag Removal | \<b\>text\</b\> → text |
| 6 | Bullet/Numbering Removal | "• item" → "item" |
| 7 | Inline Bullet Removal | 특수 불릿 기호 제거 |
| 8 | Hyphen-split Word Merge | "docu- ment" → "document" |
| 9 | Newline → Space | 줄바꿈을 공백으로 치환 |
| 10 | Whitespace Normalization | 연속 공백 → 단일 공백 |

> **Table 7.** OCR 텍스트 클리닝 파이프라인. 문서 도메인에서 빈번히 발생하는 노이즈 패턴에 대응하여 설계하였다.

---

## 3. Experimental Setup

### 3.1 Evaluation Metrics

Samsung AI Challenge의 공식 평가는 다음 세 가지 모듈의 **가중합(Weighted Sum)**으로 계산된다:

| Module | Weight | Description |
|--------|--------|-------------|
| Layout Detection | **35%** | 바운딩 박스 탐지 및 분류 정확도 |
| Reading Order | **35%** | 읽기 순서 예측 정확도 |
| OCR (Text Extraction) | **30%** | 텍스트 추출 정확도 |

각 모듈은 순차 파이프라인으로 연결되므로, **상위 단계의 오류는 하위 단계 성능에 직접적으로 전파(error propagation)**된다. 따라서 Layout Detection의 정확도가 전체 시스템 성능의 상한을 결정짓는 핵심 요소이다.

### 3.2 Datasets

| 문서 유형 | 데이터셋 | 용도 |
|-----------|----------|------|
| 세로형 문서 | Upstage DP-Bench (논문, 보고서) | 평가 |
| 가로형 문서 | 대회 제공 PPT + 자체 제작 21장 + 외부 수집 79장 | Fine-tuning + 평가 |

> **Table 8.** 실험에 사용된 데이터셋 구성.

### 3.3 Implementation Details

| Item | Specification |
|------|---------------|
| GPU | NVIDIA HGX H200 |
| CUDA | 12.1 |
| PyTorch | 2.5.1+cu121 |
| Ultralytics | 8.3.185 |
| EasyOCR | 1.7.2 |
| Training Epochs | 61 (Early Stopped from 300) |
| Training Time | ~240 min (~4 hours) |
| Total Model Size | 221 MB |

> **Table 9.** 실험 환경 및 구현 세부 사항.

---

## 4. Experiments

본 장에서는 제안 시스템의 각 모듈별 성능을 정량적으로 평가한다. 모든 실험은 Samsung AI Challenge **공식 평가 지표(Public Score)**를 기준으로 수행하였다.

### 4.1 Layout Detection Experiments

#### 4.1.1 세로형 문서: Rule-based Post-processing 효과

세로형 문서에서 사전학습된 YOLOv12l-DocLayNet의 탐지 결과에 Rule-based Post-processing을 적용한 효과를 평가하였다.

| Setting | Public Score | Improvement |
|---------|-------------|-------------|
| YOLOv12l-DocLayNet (Raw) | 0.165 | — |
| + Rule-based Post-processing | 0.245 | **+48.5%** |

> **Table 10.** 세로형 문서에서의 Rule-based Post-processing 효과. NMS, Title 재분류, Caption 필터링을 통해 약 48% 성능 향상을 달성하였다.

#### 4.1.2 가로형 문서: Fine-tuning 효과

가로형 문서에서는 사전학습 모델만으로는 성능이 불충분하였으며, 도메인 특화 Fine-tuning을 통해 유의미한 성능 향상을 달성하였다.

| Model | Public Score | Improvement |
|-------|-------------|-------------|
| YOLOv12l-DocLayNet (Base) | 0.345 | — |
| YOLOv12l-DocLayNet + Fine-tuning | 0.436 | **+26.4%** |

> **Table 11.** 가로형 문서에서의 Fine-tuning 효과. 100장의 포스터 데이터로 학습하여 약 26% 성능 향상을 달성하였다.

#### 4.1.3 Detection Confidence Optimization

Optuna를 활용하여 클래스별 confidence threshold를 대회 점수를 목적 함수로 자동 최적화하였다.

| Setting | Public Score | Improvement |
|---------|-------------|-------------|
| Before Optimization | 0.245 | — |
| After Optimization | 0.443 | **+80.8%** |

> **Table 12.** Optuna 기반 신뢰도 임계값 최적화 효과. Detection 단계 기준 약 81% 성능 향상을 기록하였다.

#### 4.1.4 Fine-tuning Training Dynamics

Fine-tuning 학습 과정에서의 성능 변화를 아래에 정리하였다.

| Epoch | Train Box Loss | Train Cls Loss | mAP50 | mAP50-95 | Precision | Recall |
|-------|---------------|----------------|-------|----------|-----------|--------|
| 1 | 1.128 | 3.720 | 0.132 | 0.100 | 0.086 | 0.453 |
| 10 | 0.595 | 0.664 | 0.789 | 0.627 | 0.735 | 0.724 |
| 20 | 0.506 | 0.421 | 0.849 | 0.692 | 0.846 | 0.770 |
| 30 | 0.468 | 0.362 | 0.849 | 0.682 | 0.806 | 0.818 |
| 40 | 0.450 | 0.335 | 0.856 | 0.699 | 0.816 | 0.818 |
| 50 | 0.434 | 0.309 | 0.849 | 0.694 | 0.819 | 0.782 |
| **61** | **0.422** | **0.292** | **0.852** | **0.701** | **0.841** | **0.809** |

> **Table 13.** Fine-tuning 학습 곡선. Epoch 30 이후 검증 메트릭이 안정화되었으며, Epoch 61에서 Early Stopping이 작동하였다. 최종 mAP50 = 0.852, mAP50-95 = 0.701을 달성하였다.

### 4.2 Reading Order Experiments

인간의 독서 습관을 모방한 세 가지 읽기 순서 전략을 점진적으로 적용하며 성능 변화를 평가하였다.

| Strategy | Description | Improvement |
|----------|-------------|-------------|
| Baseline (No order) | 탐지 순서 그대로 출력 | — |
| + Book-style | 단일 컬럼 위→아래 | **+21.4%** |
| + Newspaper-style | 멀티컬럼 감지 + 컬럼별 정렬 | **+57.1%** |
| + Poster-style (Scan-mode) | Subtitle 기반 영역 분할 | **+128.6%** |

> **Table 14.** Reading Order 전략별 누적 성능 향상. 세 가지 전략을 계층적으로 통합함으로써 총 128.6%의 성능 향상을 달성하였다.

**분석:** 가장 큰 성능 도약은 Poster-style 전략의 추가에서 발생하였다. 이는 가로형 문서(PPTX, 포스터)가 대회 평가 데이터에서 상당 비중을 차지하며, 이러한 문서에서 Subtitle 기반 영역 분할이 읽기 순서 정확도에 결정적 역할을 함을 시사한다.

### 4.3 OCR Experiments

#### 4.3.1 OCR Engine Comparison

동일 데이터셋에서 Surya OCR과 EasyOCR의 추론 시간을 비교하였다.

| OCR Engine | Inference Time | Speed Ratio |
|-----------|---------------|-------------|
| Surya OCR | 26 min 14 sec | 1.0× |
| EasyOCR | 9 min 41 sec | **2.7×** |

> **Table 15.** OCR 추론 시간 비교. EasyOCR은 Surya 대비 약 2.7배 빠른 추론 속도를 보이며, 유사한 수준의 정확도를 유지하였다.

#### 4.3.2 DPI Optimization Effect

입력 이미지의 DPI가 OCR 성능에 미치는 영향을 분석하였다.

| DPI | Observation |
|-----|-------------|
| 200 | 문자 누락 빈번, 소형 텍스트 인식 실패, 인식 오류 다수 발생 |
| 800 | 문장 단위 복원 가능, 소형 텍스트 및 수식 인식 정확도 대폭 개선 |

> **Table 16.** DPI에 따른 OCR 품질 변화. 추가 모델 학습 없이 DPI 조정만으로 OCR 성능을 크게 향상시킬 수 있음을 확인하였다.

---

## 5. Results

### 5.1 End-to-End Performance

Detection, Reading Order, OCR을 모두 통합한 최종 End-to-End 성능을 평가하였다.

| System | Public Score | Improvement |
|--------|-------------|-------------|
| Baseline | 0.1168 | — |
| **Proposed Method** | **0.4577** | **+292%** |

> **Table 17.** End-to-End 성능 비교. 제안 시스템은 Baseline 대비 292%의 성능 향상을 달성하였다.

### 5.2 Module-wise Contribution Analysis

각 모듈의 기여도를 분석하면 다음과 같다:

| Module | Key Improvement | Main Factor |
|--------|----------------|-------------|
| Layout Detection | +48% (후처리) / +26% (Fine-tuning) / +81% (Optuna) | 적응형 모델 선택 + 후처리 + 임계값 최적화 |
| Reading Order | +128.6% (전략 통합) | Poster-style Scan-mode 알고리즘 |
| OCR | 2.7× 속도 향상 | EasyOCR + DPI 800 최적화 |

### 5.3 System Efficiency

| Metric | Value |
|--------|-------|
| Total Model Size | **221 MB** |
| OCR Speed Improvement | **2.7×** (vs Surya) |
| Training Time | ~4 hours (NVIDIA H200) |
| Execution Mode | **Fully On-device** |

> **Table 18.** 시스템 효율성 지표. 221MB의 경량 모델 크기와 On-device 실행이 가능한 구조로, 실용적 배포 환경에 적합하다.

### 5.4 Fine-tuning Model Performance Summary

| Metric | Value |
|--------|-------|
| Precision (B) | 0.8415 |
| Recall (B) | 0.8089 |
| mAP50 (B) | **0.8523** |
| mAP50-95 (B) | **0.7008** |
| Training Epochs | 61 / 300 (Early Stopped) |
| Convergence | Epoch 30 이후 안정화 |

> **Table 19.** Fine-tuned 모델의 최종 성능. 100장의 학술 포스터로 학습하여 mAP50 = 0.852를 달성하였다.

---

## 6. Conclusion

본 연구에서는 Visually-rich Document Understanding 문제를 On-device 환경에서 해결하기 위한 **통합 문서 이해 시스템**을 제안하였다.

### 주요 성과

1. **적응형 이중 모델 전략**을 통해 세로형·가로형 문서의 구조적 차이를 효과적으로 반영하였으며, 가로형 문서에서 Fine-tuning을 통해 **26%의 탐지 성능 향상**을 달성하였다.

2. **규칙 기반 후처리**를 통해 반복적 오류 패턴(중복 탐지, Title 오분류, Caption 오탐)을 교정하여 세로형 문서에서 **48%의 추가 성능 향상**을 확보하였다.

3. **Optuna 기반 신뢰도 최적화**를 통해 클래스별 confidence threshold를 자동 최적화하여 Detection 단계에서 **81%의 성능 향상**을 기록하였다.

4. 인간의 독서 습관을 모방한 **세 가지 구조 기반 읽기 순서 알고리즘**(책·신문·포스터 읽기)을 계층적으로 통합하여 Reading Order에서 **128.6%의 성능 향상**을 달성하였다.

5. EasyOCR과 DPI 최적화를 통해 Surya OCR 대비 **2.7배 빠른 추론 속도**를 확보하면서도 정확도를 유지하였다.

6. 최종적으로, 제안 시스템은 Baseline 대비 **Public Score 292% 향상**(0.1168 → 0.4577)을 달성하였으며, **221MB의 경량 모델 크기**로 On-device 배포가 가능하다.

### 향후 연구 방향

- **표(Table) 내부 구조 분석**: 셀 단위 인식 및 구조화된 데이터 추출
- **다양한 문서 유형 확장**: 영수증, 청구서, 법률 문서 등 도메인 확장
- **Transformer 기반 레이아웃 모델 비교**: LayoutLMv3, DiT 등과의 성능 비교
- **학습 데이터 규모 확대**: 현재 100장 → 대규모 다양한 문서 데이터셋 구축
- **읽기 순서 정량적 평가 메트릭 설계**: Ground Truth 기반 Reading Order 평가 체계 개발

---

## References

1. Ultralytics. "YOLOv12: Real-time Object Detection." AGPL-3.0. https://github.com/ultralytics/ultralytics
2. B. Pfitzmann, C. Auer, M. Dolfi, A. S. Nassar, and P. Staar. "DocLayNet: A Large Human-Annotated Dataset for Document-Layout Segmentation." *Proceedings of KDD*, 2022.
3. Y. Baek, B. Lee, D. Han, S. Yun, and H. Lee. "Character Region Awareness for Text Detection." *Proceedings of CVPR*, 2019.
4. JaidedAI. "EasyOCR: Ready-to-use OCR with 80+ Supported Languages." Apache-2.0. https://github.com/JaidedAI/EasyOCR
5. T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama. "Optuna: A Next-generation Hyperparameter Optimization Framework." *Proceedings of KDD*, 2019.
6. Roboflow. "Computer Vision Dataset Management Platform." https://roboflow.com

---

## Appendix

### A. LaTeX 섹션 매핑

본 문서의 각 섹션은 LaTeX 템플릿의 `\input{sections/...}` 파일에 다음과 같이 대응된다:

| 본 문서 섹션 | LaTeX 파일 | 비고 |
|-------------|-----------|------|
| Abstract | `sections/abstract.tex` | 초록 |
| 1. Introduction | `sections/related_work.tex` 또는 별도 `sections/introduction.tex` | 서론 + 관련 연구 |
| 2. Methodology | `sections/methodology.tex` | 핵심 방법론 |
| 3. Experimental Setup | `sections/dataset.tex` + `sections/experiments.tex` | 데이터셋 + 실험 설정 |
| 4. Experiments | `sections/experiments.tex` | 실험 결과 |
| 5. Results | `sections/results.tex` | 종합 결과 |
| 6. Conclusion | `sections/conclusion.tex` | 결론 |
| References | `sections/references.tex` | 참고문헌 |

### B. 전체 하이퍼파라미터 요약

```
# Detection
CONF_TITLE (Base)     = 0.15    CONF_TITLE (Finetune)    = 0.45
CONF_SUBTITLE (Base)  = 0.15    CONF_SUBTITLE (Finetune) = 0.10
CONF_TEXT (Base)       = 0.10    CONF_TEXT (Finetune)     = 0.10
CONF_STRUCT (Base)    = 0.10    CONF_IMAGE (Finetune)    = 0.2246

# Post-processing
NMS_IOU_THRESHOLD                = 0.5
SUBTITLE_PROMOTE_TO_TITLE_CONF  = 0.80
CAPTION_FILTER_MIN_CONF          = 0.95

# Reading Order
SPAN_WIDTH_RATIO     = 0.65     MAX_COLS              = 3
Y_TOL_RATIO          = 0.15     COL_GAP_RATIO(2)      = 0.30
MIN_LINE_TOL_PX      = 8        COL_GAP_RATIO(3)      = 0.22
ROW_Y_TOL_FACTOR     = 0.6      ROW_OVERLAP_THR       = 0.35

# OCR
DPI                  = 800
Languages            = ['ko', 'en']

# Training
EPOCHS = 300 (early stop 61)    BATCH = 16
IMGSZ  = 1024                   LR0   = 0.01
PATIENCE = 50                   WEIGHT_DECAY = 0.0005
```

### C. 성능 향상 요약 (한눈에 보기)

```
                        Public Score
Baseline               ████                          0.1168
+ Rule-based PP        ██████                        0.245   (+110%)
+ Fine-tuning          ████████████                  0.436   (+78%)
+ Optuna Conf Opt      █████████████                 0.443   (+2%)
+ Reading Order        ██████████████████            0.457   (+3%)
+ OCR (800 DPI)        ██████████████████            0.4577  (final)
─────────────────────────────────────────────────────────────
Total Improvement       Baseline → Final: +292%
```
