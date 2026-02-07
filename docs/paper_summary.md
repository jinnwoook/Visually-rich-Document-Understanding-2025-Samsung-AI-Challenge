# Visually-rich Document Understanding을 위한 멀티모달 딥러닝 접근법
## 2025 Samsung AI Challenge 참가 보고서 - 논문 작성용 정리

---

## 1. Abstract (초록)

본 연구는 2025 삼성 AI 챌린지의 **Visually-rich Document Understanding** 과제에 대한 솔루션을 제시한다.
PDF, PPTX, 이미지 등 다양한 형식의 문서에서 **(1) 레이아웃 요소 탐지(Layout Detection)**, **(2) 요소 분류(Element Classification)**, **(3) 읽기 순서 예측(Reading Order Prediction)**, **(4) 텍스트 추출(OCR)**을 통합적으로 수행하는 멀티모달 파이프라인을 설계하였다.

**핵심 접근법:**
- **YOLOv12-DocLayNet** 기반 문서 레이아웃 탐지 + 도메인 특화 Fine-tuning
- **이중 모델 전략(Dual-Model Strategy)**: 세로형(PDF) vs 가로형(PPTX) 문서에 대한 분리 처리
- **스캔 모드 읽기 순서 알고리즘**: 멀티컬럼 감지 + subtitle 기반 영역 분할
- **EasyOCR** 기반 한/영 이중언어 텍스트 추출
- **NVIDIA HGX H200** 환경에서 학습, mAP50 = **0.852** 달성

---

## 2. Related Work (관련 연구)

### 2.1 문서 레이아웃 분석 (Document Layout Analysis)
| 모델/기술 | 역할 | 라이선스 |
|-----------|------|----------|
| **YOLOv12** (Ultralytics) | 실시간 객체 탐지 프레임워크 | AGPL-3.0 |
| **DocLayNet** (IBM) | 문서 레이아웃 사전학습 데이터셋 | - |
| **YOLOv12-DocLayNet** | DocLayNet으로 사전학습된 YOLO v12 Large | AGPL-3.0 |

### 2.2 텍스트 탐지 및 인식 (OCR)
| 모델/기술 | 역할 | 라이선스 |
|-----------|------|----------|
| **CRAFT** (CLOVA AI) | 문자 영역 인식 기반 텍스트 탐지 | Apache-2.0 |
| **EasyOCR** (JaidedAI) | 다국어(한/영) 텍스트 인식 | Apache-2.0 |

### 2.3 읽기 순서 예측 (Reading Order)
- KMeans 클러스터링 기반 **멀티컬럼 자동 감지**
- Subtitle 행(row) 그룹핑 기반 **영역 분할 알고리즘**

---

## 3. Dataset (데이터셋)

### 3.1 학습 데이터셋 구성
| 항목 | 내용 |
|------|------|
| **데이터 소스** | NeurIPS 2024 학술 포스터 (Roboflow 수집) |
| **총 이미지 수** | 100장 |
| **라이선스** | CC BY 4.0 |
| **어노테이션 형식** | YOLO Detection Format (bbox + class) |
| **이미지 해상도** | 1123x794 ~ 5120x2880 (가변) |

### 3.2 클래스 정의 (6 Classes)
| 클래스 | YOLO ID | 설명 |
|--------|---------|------|
| `equation` | 0 | 수식 (Formula) |
| `image` | 1 | 이미지/그림 (Picture) |
| `subtitle` | 2 | 섹션 헤더 (Section-header) |
| `table` | 3 | 표 (Table) |
| `text` | 4 | 본문 텍스트 (Body text) |
| `title` | 5 | 제목 (Title) |

### 3.3 데이터 분할
| 분할 | 비율 | 이미지 수 |
|------|------|-----------|
| Training | 90% | 90장 |
| Validation | 10% | 10장 |
| Test | 0% | 0장 |

### 3.4 데이터 출처
- **Roboflow Workspace**: `new-workspace-t7umm`
- **Project**: `ppt-dogsc` (Version 5)
- **URL**: https://universe.roboflow.com/new-workspace-t7umm/ppt-dogsc/dataset/5

---

## 4. Methodology (방법론)

### 4.1 전체 파이프라인 아키텍처

```
입력 문서 (PDF/PPTX/이미지)
        │
        ▼
┌─────────────────────┐
│ 1. 문서 변환         │  PDF → 800 DPI PNG
│    (Document Convert)│  PPTX → LibreOffice → PDF → PNG
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ 2. 문서 유형 판별     │  가로형(landscape) vs 세로형(portrait)
│    (Layout Branch)   │  PPTX 여부 확인
└─────────────────────┘
        │
        ├── 가로형/PPTX ──→ Fine-tuned Model (V5.pt)
        │
        └── 세로형/PDF  ──→ Base Model (yolov12l-doclaynet.pt)
        │
        ▼
┌─────────────────────┐
│ 3. 전역 탐지         │  이미지 1280x1280 리사이즈
│    (Global Detection)│  YOLO v12 추론 (imgsz=1024)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ 4. 후처리            │  NMS (IoU 0.5)
│    (Post-Processing) │  Subtitle→Title 승격
│                      │  단일 Top-Title 강제
│                      │  Caption 필터링
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ 5. OCR 텍스트 추출    │  EasyOCR (한국어+영어)
│    (Text Extraction) │  텍스트 클리닝 파이프라인
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ 6. 읽기 순서 결정     │  세로형: 문서 유형 분류 → 멀티컬럼 감지
│    (Reading Order)   │  가로형: Scan-mode (subtitle 행 기반)
└─────────────────────┘
        │
        ▼
    최종 CSV 출력
```

### 4.2 이중 모델 전략 (Dual-Model Strategy)

**핵심 통찰**: 세로형 문서(논문 PDF 등)와 가로형 문서(PPTX 프레젠테이션)는 레이아웃 특성이 근본적으로 다르므로, 각각에 최적화된 모델을 사용한다.

```python
if document_type == PPTX or target_width > target_height:  # 가로형
    model = model_finetune  # V5.pt (NeurIPS 포스터로 fine-tuned)
else:                                                        # 세로형
    model = model_doc       # yolov12l-doclaynet.pt (DocLayNet 사전학습)
```

**모델별 신뢰도 임계값:**

| 카테고리 | Base Model (세로형) | Fine-tuned Model (가로형) |
|----------|---------------------|---------------------------|
| Title | 0.15 | 0.45 |
| Subtitle | 0.15 | 0.10 |
| Text | 0.10 | 0.10 |
| Table | 0.10 | 0.10 |
| Equation | 0.10 | 0.10 |
| Image | 0.10 | 0.2246 |

### 4.3 Fine-tuning 전략

#### 4.3.1 베이스 모델
- **모델**: YOLOv12-Large (DocLayNet 사전학습)
- **출처**: `hantian/yolo-doclaynet` (HuggingFace)
- **크기**: ~51 MB

#### 4.3.2 학습 하이퍼파라미터
| 파라미터 | 값 | 설명 |
|----------|-----|------|
| Input Size | 1024x1024 | Letterbox 리사이즈 |
| Batch Size | 16 | - |
| Max Epochs | 300 | Early stopping 적용 |
| Patience | 50 | 50 에폭 동안 개선 없으면 종료 |
| Optimizer | Auto (SGD) | Ultralytics 자동 선택 |
| Initial LR | 0.01 | - |
| Final LR ratio | 0.01 | Cosine annealing |
| LR Scheduler | **Cosine Annealing** | cos_lr=True |
| Warmup | 3 에폭 | LR 0.0001 → 0.01 |
| Weight Decay | 0.0005 | - |
| Momentum | 0.937 | SGD momentum |
| AMP | True | Mixed Precision Training |
| Seed | 42 | 재현성 보장 |
| Deterministic | True | cuDNN 결정적 모드 |

#### 4.3.3 데이터 증강 전략

**문서 도메인 특화 보수적 증강**: 텍스트의 가독성과 문서 구조를 보존하기 위해 기하학적 변환을 최소화함.

| 증강 기법 | 값 | 사용 여부 | 이유 |
|-----------|-----|-----------|------|
| **Mosaic** | 0.2 (20%) | O (제한적) | 마지막 10 에폭 비활성화 |
| **수평 뒤집기** | 0.0 | X | 텍스트 방향 보존 |
| **수직 뒤집기** | 0.0 | X | 문서 방향 보존 |
| **회전** | 0° | X | 문서 정렬 유지 |
| **이동** | ±5% | O (미세) | 위치 일반화 |
| **스케일** | ±20% | O | 다양한 해상도 대응 |
| HSV Hue | 0.0 | X | 색상 불변 |
| HSV Saturation | 0.30 | O | 경미한 색 변화 |
| HSV Value | 0.20 | O | 경미한 밝기 변화 |
| **Erasing (Cutout)** | 0.4 | O | 부분 가림에 대한 강건성 |
| MixUp | 0.0 | X | - |
| CutMix | 0.0 | X | - |
| Copy-Paste | 0.0 | X | - |

### 4.4 후처리 파이프라인 (Post-Processing)

#### 4.4.1 NMS (Non-Maximum Suppression)
- IoU 임계값: **0.5**
- 신뢰도 높은 순으로 정렬 후 중복 박스 제거

#### 4.4.2 분류 보정 (Classification Refinement)
1. **Subtitle → Title 승격**: Base 모델 사용 시, subtitle의 신뢰도가 0.80 미만이면 title로 승격
2. **단일 Top-Title 강제**: 복수 title이 탐지되면, 가장 위쪽·왼쪽의 것만 title로 유지하고 나머지는 subtitle로 변경
3. **Caption 필터링**: "Figure X:", "Table X:", "그림", "표" 등 캡션 패턴의 텍스트 박스를 신뢰도 0.95 미만일 때 제거

#### 4.4.3 False Table 감지 파라미터 (선언됨, 미사용)
- 면적 비율 상한: 0.35
- 너비/높이 비율 상한: 0.95/0.85
- 내부 텍스트 박스 최소 수: 4

### 4.5 읽기 순서 알고리즘 (Reading Order)

#### 4.5.1 세로형 문서 (PDF/Portrait)

```
1. 문서 유형 분류 (paper vs general)
   - Text 스팬 탐지: 페이지 너비의 65% 이상인 텍스트 박스
   - 스팬이 3개 이상 → "general", 미만 → "paper"

2. "paper" 유형:
   a. 텍스트 스팬을 y좌표 기준 정렬
   b. 스팬 사이 영역을 추출
   c. 각 영역에서 멀티컬럼 감지 (KMeans)
   d. 컬럼별 위→아래, 왼쪽→오른쪽 순서
   e. 스팬은 발견 순서에 따라 삽입

3. "general" 유형:
   a. 모든 항목을 y좌표 → x좌표로 정렬
   b. 라인 그룹핑 (y-tolerance 기반)
   c. 각 라인 내 좌→우 정렬
```

#### 4.5.2 가로형 문서 (PPTX/Landscape) - Scan-Mode

```
1. Subtitle 행(Row) 그룹핑
   - 수직 위치가 유사한 subtitle들을 같은 행으로 묶음
   - Row tolerance: max(24px, median_height × 0.6)
   - Vertical overlap 임계값: 0.35

2. y↑ 방향 스캔
   FOR EACH item (y좌표 오름차순):
     IF subtitle:
       해당 행의 모든 subtitle 처리
       - 숫자 접두사 2개↑ → 숫자순 정렬 (예: "1. 서론", "2. 방법")
       - 그 외 → 좌→우 정렬

       각 subtitle의 세로 스트립 처리:
       - 좌우 경계: 인접 subtitle 중심점의 중간
       - 상하 경계: subtitle 하단 ~ 다음 행 시작
       - 스트립 내부: 멀티컬럼 감지 → 논문 읽기 방식

     ELSE (비-subtitle):
       다음 subtitle 행까지의 영역을 통합 레이아웃 로직으로 처리
```

#### 4.5.3 멀티컬럼 감지 알고리즘
```python
# KMeans 클러스터링으로 2~3컬럼 자동 감지
max_columns = 3
for k in [3, 2]:  # 높은 컬럼 수부터 시도
    centers = KMeans(n_clusters=k).fit(center_x_positions)
    gaps = diff(sorted_centers)

    # 컬럼 간 간격이 충분히 넓은지 검증
    gap_threshold = {2: 30%, 3: 22%} × page_width
    min_items_per_col = {2: 2, 3: 2}

    if all(gaps >= threshold) and all(col_count >= min_items):
        return columns  # 멀티컬럼 레이아웃 확인
return None  # 단일 컬럼
```

#### 4.5.4 라인 그룹핑
- Y-tolerance: `max(8px, median_box_height × 15%)`
- 같은 수평선상의 요소들을 하나의 라인으로 묶음
- 라인 내부: x좌표 기준 좌→우 정렬

### 4.6 OCR 텍스트 추출

#### 4.6.1 OCR 파이프라인
```
탐지된 텍스트 영역 → 바운딩 박스 crop → EasyOCR 추론 → 텍스트 클리닝
```

#### 4.6.2 EasyOCR 설정
- **지원 언어**: 한국어(ko) + 영어(en)
- **모델**: CRAFT 텍스트 탐지 + 한/영 인식 모델
- **GPU 가속**: CUDA 사용 가능 시 자동 활성화

#### 4.6.3 텍스트 클리닝 파이프라인
| 단계 | 처리 내용 |
|------|-----------|
| 1 | NFKC 유니코드 정규화 |
| 2 | 줄바꿈 통일 (`\r\n` → `\n`) |
| 3 | 수학/스크립트 태그 제거 (`<MATH>`, `<SCRIPT>`) |
| 4 | 하이픈 줄바꿈 병합 (`hyphen-\nword` → `hyphenword`) |
| 5 | HTML 태그 제거 (`<b>`, `<span>` 등) |
| 6 | 불릿 포인트/번호 매기기 제거 (•, ①, 1. 등) |
| 7 | 인라인 불릿 기호 제거 |
| 8 | 하이픈 분리 단어 병합 |
| 9 | 줄바꿈 → 공백 치환 |
| 10 | 연속 공백 정리 |

### 4.7 입력 문서 처리

| 입력 형식 | 변환 방법 | DPI |
|-----------|-----------|-----|
| PDF | `pdf2image` (poppler) | 800 |
| PPTX | LibreOffice → PDF → PNG | 800 |
| JPG/PNG | 직접 로딩 (PIL) | - |

---

## 5. Experiments (실험)

### 5.1 실험 환경
| 항목 | 사양 |
|------|------|
| **GPU** | NVIDIA HGX H200 |
| **CUDA** | 12.1 |
| **PyTorch** | 2.5.1+cu121 |
| **Ultralytics** | 8.3.185 |
| **Python** | 3.x (conda 환경) |
| **EasyOCR** | 1.7.2 |
| **scikit-learn** | 1.7.1 |

### 5.2 학습 과정
| 항목 | 수치 |
|------|------|
| 설정 에폭 | 300 |
| 실제 학습 에폭 | **61** (Early Stopping) |
| 총 학습 시간 | **~240분 (4시간)** |
| 에폭 당 시간 | ~3.9분 |
| 수렴 시점 | ~30 에폭 (이후 안정화) |

### 5.3 학습 곡선 주요 지점

| 에폭 | Box Loss | Cls Loss | mAP50 | mAP50-95 | Precision | Recall |
|------|----------|----------|-------|----------|-----------|--------|
| 1 | 1.128 | 3.720 | 0.132 | 0.100 | 0.086 | 0.453 |
| 10 | 0.595 | 0.664 | 0.789 | 0.627 | 0.735 | 0.724 |
| 20 | 0.506 | 0.421 | 0.849 | 0.692 | 0.846 | 0.770 |
| 30 | 0.468 | 0.362 | 0.849 | 0.682 | 0.806 | 0.818 |
| 40 | 0.450 | 0.335 | 0.856 | 0.699 | 0.816 | 0.818 |
| 50 | 0.434 | 0.309 | 0.849 | 0.694 | 0.819 | 0.782 |
| **61** (최종) | **0.422** | **0.292** | **0.852** | **0.701** | **0.841** | **0.809** |

### 5.4 Learning Rate 스케줄
```
Epoch 1-3: Warmup (0.00005 → 0.01)
Epoch 4-61: Cosine Annealing (0.01 → 0.0009)
```

---

## 6. Results (결과)

### 6.1 최종 성능 메트릭

| 메트릭 | 값 |
|--------|-----|
| **Precision(B)** | 0.8415 |
| **Recall(B)** | 0.8089 |
| **mAP50(B)** | **0.8523** |
| **mAP50-95(B)** | **0.7008** |
| Train Box Loss | 0.4223 |
| Train Cls Loss | 0.2921 |
| Val Box Loss | 0.8755 |
| Val Cls Loss | 0.8486 |

### 6.2 성능 향상 추이
- **Epoch 1 → 61**: mAP50 0.132 → 0.852 (**6.5배 향상**)
- **Epoch 1 → 61**: mAP50-95 0.100 → 0.701 (**7.0배 향상**)
- **Epoch 30 이후**: 검증 손실 안정화, 과적합 징후 없음

### 6.3 모델 체크포인트
| 체크포인트 | 용도 | 크기 |
|-----------|------|------|
| `best.pt` | 최고 검증 성능 모델 | 51 MB |
| `last.pt` | 최종 에폭 모델 | 51 MB |
| `V5.pt` | 추론용 fine-tuned 모델 | 51 MB |

### 6.4 모델 아키텍처 비교
| 모델 | 용도 | 사전학습 | Fine-tuning | 대상 문서 |
|------|------|----------|-------------|-----------|
| yolov12l-doclaynet.pt | 세로형 문서 탐지 | DocLayNet | X | PDF, 논문 |
| V5.pt | 가로형 문서 탐지 | DocLayNet | NeurIPS 포스터 100장 | PPTX, 포스터 |

---

## 7. 핵심 전략 및 기술적 기여 (Key Strategies & Contributions)

### 7.1 이중 모델 전략
- **문제 인식**: 세로형(논문 PDF)과 가로형(프레젠테이션) 문서는 레이아웃 특성이 근본적으로 다름
- **해결책**: 문서 방향에 따라 다른 모델과 임계값 자동 선택
- **효과**: 각 도메인에 최적화된 탐지 성능 확보

### 7.2 도메인 특화 Fine-tuning
- **데이터**: NeurIPS 2024 학술 포스터 100장으로 가로형 문서에 특화
- **보수적 증강**: 텍스트 가독성 보존을 위해 뒤집기/회전 비활성화
- **효과**: 가로형 문서에서의 레이아웃 탐지 정확도 대폭 향상

### 7.3 스캔 모드 읽기 순서
- **혁신**: subtitle 행(row)을 기준으로 문서를 영역 분할하는 새로운 읽기 순서 알고리즘
- **숫자 접두사 자동 인식**: "1. 서론", "2. 방법론" 등 번호 매기기 패턴 자동 감지 및 순서 정렬
- **멀티컬럼 감지**: KMeans 클러스터링으로 2~3컬럼 레이아웃 자동 인식
- **효과**: 복잡한 포스터/프레젠테이션 문서의 읽기 순서 정확도 향상

### 7.4 강건한 텍스트 클리닝
- **다단계 정규화**: NFKC 유니코드, HTML 태그, 수학 표기, 불릿 포인트 등 체계적 제거
- **캡션 필터링**: "Figure X:", "Table X:" 등 캡션 패턴 자동 탐지 및 제거
- **효과**: OCR 결과의 노이즈 최소화

### 7.5 높은 해상도 입력
- **PDF → PNG 변환**: 800 DPI 고해상도
- **YOLO 입력**: 1280x1280 리사이즈 후 1024 크기로 추론
- **효과**: 작은 텍스트와 수식도 정확하게 탐지

---

## 8. 출력 형식 (Output Format)

### CSV 출력 스키마
```
ID, category_type, confidence_score, order, text, bbox
```

### 예시
```csv
doc1_p1, title, 0.95, 0, "Attention Is All You Need", "10, 20, 500, 80"
doc1_p1, text, 0.88, 1, "We propose a new simple network...", "10, 100, 500, 300"
doc1_p1, image, 0.91, 2, "", "50, 320, 450, 550"
```

---

## 9. 의존성 및 환경 (Dependencies)

### 핵심 라이브러리
| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| ultralytics | 8.3.185 | YOLOv12 프레임워크 |
| torch | 2.5.1+cu121 | 딥러닝 백엔드 |
| torchvision | 0.20.1+cu121 | 비전 유틸리티 |
| easyocr | 1.7.2 | 다국어 OCR |
| pdf2image | 1.17.0 | PDF→이미지 변환 |
| Pillow | 11.3.0 | 이미지 처리 |
| pandas | 2.3.2 | CSV 데이터 핸들링 |
| numpy | 2.2.6 | 수치 연산 |
| scikit-learn | 1.7.1 | KMeans 클러스터링 |
| opencv-python-headless | 4.12.0.88 | 컴퓨터 비전 |
| regex | 2025.7.34 | 고급 정규식 |

---

## 10. Conclusion (결론)

본 연구에서는 Visually-rich Document Understanding 문제를 해결하기 위한 종합적인 멀티모달 파이프라인을 제시하였다.

**주요 성과:**
1. YOLOv12-DocLayNet 기반 문서 레이아웃 탐지에서 mAP50 **0.852**, mAP50-95 **0.701** 달성
2. 이중 모델 전략을 통해 세로형/가로형 문서 모두에 대응하는 유연한 시스템 구축
3. 스캔 모드 읽기 순서 알고리즘으로 복잡한 멀티컬럼 문서의 읽기 순서 정확도 향상
4. 100장의 학술 포스터로 fine-tuning하여 가로형 문서 탐지 성능 극대화
5. EasyOCR 기반 한/영 이중언어 텍스트 추출 파이프라인 구축

**향후 개선 방향:**
- 더 큰 규모의 다양한 문서 데이터셋 구축
- 표(table) 내부 구조 분석 추가
- Transformer 기반 레이아웃 모델 (LayoutLMv3 등) 비교 실험
- 읽기 순서에 대한 Ground Truth 기반 정량적 평가 메트릭 설계

---

## 11. References (참고문헌)

1. **YOLOv12**: Ultralytics YOLO v12, AGPL-3.0, https://github.com/ultralytics/ultralytics
2. **DocLayNet**: Pfitzmann et al., "DocLayNet: A Large Human-Annotated Dataset for Document-Layout Segmentation", KDD 2022
3. **CRAFT**: Baek et al., "Character Region Awareness for Text Detection", CVPR 2019
4. **EasyOCR**: JaidedAI, https://github.com/JaidedAI/EasyOCR, Apache-2.0
5. **Roboflow**: Dataset Platform, https://roboflow.com
6. **KMeans Clustering**: scikit-learn, https://scikit-learn.org

---

## 12. Appendix (부록)

### A. 레이블 매핑 테이블
```python
LABEL_MAP = {
    'Text': 'text',           'Title': 'title',
    'Section-header': 'subtitle',  'Formula': 'equation',
    'Table': 'table',         'Picture': 'image',
}
```

### B. 추론 신뢰도 임계값 전체
```python
# Base Model (세로형)
CONF_TITLE    = 0.15
CONF_SUBTITLE = 0.15
CONF_TEXT     = 0.10
CONF_STRUCT   = 0.10

# Fine-tuned Model (가로형)
ROI_CONF_TITLE    = 0.45
ROI_CONF_SUBTITLE = 0.10
ROI_CONF_TEXT     = 0.10
ROI_CONF_TABLE    = 0.10
ROI_CONF_EQUATION = 0.10
ROI_CONF_IMAGE    = 0.2246
```

### C. 읽기 순서 파라미터
```python
SPAN_WIDTH_RATIO = 0.65          # 스팬 판정 비율
Y_TOL_RATIO = 0.15               # 라인 그룹핑 y-tolerance
MIN_LINE_TOL_PX = 8              # 최소 라인 tolerance
MAX_COLS = 3                      # 최대 컬럼 수
COL_GAP_RATIO = {2: 0.30, 3: 0.22}  # 컬럼 간격 비율
ROW_Y_TOL_FACTOR = 0.6           # Subtitle 행 그룹핑 tolerance
ROW_OVERLAP_THR = 0.35           # 수직 오버랩 임계값
```

### D. 프로젝트 디렉토리 구조
```
├── finetune/                    # 학습 코드
│   ├── fine.py                  # Fine-tuning 스크립트
│   ├── cfg/data.yaml            # 데이터셋 설정
│   ├── data/                    # 학습 이미지/라벨
│   ├── models/base/             # 사전학습 모델
│   └── train_6cls_1024/         # 학습 결과
│       ├── weights/best.pt      # 최고 성능 체크포인트
│       ├── results.csv          # 학습 로그
│       └── args.yaml            # 하이퍼파라미터
├── gototop/                     # 추론 코드
│   ├── script.py                # 전체 추론 파이프라인 (798줄)
│   └── model/                   # 추론용 모델 파일
├── finetuning_dataset/          # 학습 데이터셋 메타정보
├── condalist.txt                # 환경 정보 (GPU, 라이브러리)
└── ReadMe.txt                   # 프로젝트 설명
```

### E. LaTeX 논문 섹션 매핑

| MD 섹션 | LaTeX 섹션 파일 |
|---------|----------------|
| 1. Abstract | `sections/abstract.tex` |
| 2. Related Work | `sections/related_work.tex` |
| 3. Dataset | `sections/dataset.tex` |
| 4. Methodology | `sections/methodology.tex` |
| 5. Experiments | `sections/experiments.tex` |
| 6. Results | `sections/results.tex` |
| 7-8. Strategies + Output | `sections/conclusion.tex` |
| 9. Dependencies | `sections/appendix.tex` |
| 11. References | `sections/references.tex` |
