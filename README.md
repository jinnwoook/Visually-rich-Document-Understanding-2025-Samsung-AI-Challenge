# Visually-rich Document Understanding

### 2025 Samsung AI Challenge | On-device Document Understanding System

> PDF, PPTX, 이미지 등 다양한 형식의 문서에서 **레이아웃 탐지**, **읽기 순서 예측**, **텍스트 추출(OCR)**을 통합 수행하는 멀티모달 문서 이해 파이프라인

---

## Overview

본 프로젝트는 **2025 삼성 AI 챌린지 Visually-rich Document Understanding** 트랙에 참가하여 개발한 On-device 문서 이해 시스템입니다.

기존 OCR 기술은 단순 텍스트 추출에 머물러, 문서의 **구조적 레이아웃**과 **읽기 흐름**을 충분히 반영하지 못합니다. 본 시스템은 이를 해결하기 위해 **YOLOv12 기반 레이아웃 탐지**, **구조 기반 읽기 순서 알고리즘**, **EasyOCR 텍스트 추출**을 하나의 파이프라인으로 통합하였습니다.

### Key Results

| Metric | Value |
|--------|-------|
| **Public Score** | **0.4577** (Baseline 대비 **+292%**) |
| Fine-tuned mAP50 | 0.852 |
| OCR 속도 개선 | Surya 대비 **2.7x** |
| 총 모델 크기 | **221 MB** (On-device 배포 가능) |

---

## Pipeline Architecture

```
입력 문서 (PDF / PPTX / Image)
        |
        v
+--------------------------------------+
|  Stage 1. Document Conversion        |
|  PDF -> 800 DPI PNG (pdf2image)      |
|  PPTX -> LibreOffice -> PDF -> PNG   |
+--------------------------------------+
        |
        v
+--------------------------------------+
|  Stage 2. Adaptive Model Selection   |
|  세로형 -> YOLOv12-DocLayNet (Base)  |
|  가로형 -> YOLOv12 Fine-tuned (V5)   |
+--------------------------------------+
        |
        v
+--------------------------------------+
|  Stage 3. Layout Detection           |
|  Global Detection (1280x1280)        |
|  + Rule-based Post-processing        |
|  + NMS / Title 재분류 / Caption 필터 |
+--------------------------------------+
        |
        v
+--------------------------------------+
|  Stage 4. Reading Order Prediction   |
|  세로형: Book / Newspaper 읽기       |
|  가로형: Poster 읽기 (Scan-mode)     |
+--------------------------------------+
        |
        v
+--------------------------------------+
|  Stage 5. OCR Text Extraction        |
|  EasyOCR (한국어 + 영어)             |
|  + 10단계 Text Cleaning Pipeline     |
+--------------------------------------+
        |
        v
    최종 출력 (CSV)
```

---

## Core Strategies

### 1. Adaptive Dual-Model Strategy (적응형 이중 모델 전략)

세로형 문서(논문, 보고서)와 가로형 문서(PPT, 포스터)는 레이아웃 특성이 근본적으로 다릅니다.
문서 방향에 따라 최적화된 모델을 **자동 선택**합니다.

| 문서 유형 | 조건 | 사용 모델 |
|-----------|------|-----------|
| 세로형 (PDF, 논문) | `width <= height` | YOLOv12-DocLayNet (Base) |
| 가로형 (PPTX, 포스터) | `width > height` 또는 PPTX | YOLOv12 Fine-tuned (V5.pt) |

### 2. Structure-based Reading Order (구조 기반 읽기 순서)

인간의 독서 습관을 모방한 **3가지 읽기 전략**을 문서 유형에 따라 계층적으로 적용합니다.

| 전략 | 적용 대상 | 방식 | 성능 향상 |
|------|-----------|------|-----------|
| **Book-style** | 단일 컬럼 문서 | 위 -> 아래, 좌 -> 우 | +21.4% |
| **Newspaper-style** | 멀티 컬럼 문서 | KMeans 컬럼 감지 -> 컬럼별 정렬 | +57.1% |
| **Poster-style (Scan-mode)** | 가로형 포스터/PPT | Subtitle 행 기반 영역 분할 | **+128.6%** |

### 3. Rule-based Post-processing (규칙 기반 후처리)

반복적 오탐 패턴을 분석하여 4단계 후처리를 적용합니다.

| 규칙 | 설명 |
|------|------|
| NMS | IoU 0.5 기반 중복 박스 제거 |
| Subtitle -> Title 승격 | 낮은 신뢰도(< 0.80)의 Subtitle을 Title로 재분류 |
| Single Top-Title | 페이지당 Title 1개 제한 |
| Caption Filtering | "Figure X:", "Table X:" 패턴 자동 제거 |

---

## Dataset

> 학습 데이터셋은 **자체 구축**하였으며, 외부 데이터와 자체 제작 데이터를 결합하여 구성하였습니다.

| 항목 | 내용 |
|------|------|
| **데이터 소스** | NeurIPS 2024 학술 포스터 |
| **구성** | 대회 제공 PPT + 자체 제작 포스터 21장 + 외부 수집 79장 |
| **총 이미지** | 100장 |
| **어노테이션** | Roboflow를 활용한 YOLO Detection Format |
| **라이선스** | CC BY 4.0 |
| **분할** | Train 90장 (90%) / Validation 10장 (10%) |

### Detection Classes (6 Classes)

| Class | ID | Description |
|-------|----|-------------|
| `title` | 5 | 문서 제목 |
| `subtitle` | 2 | 섹션 헤더 |
| `text` | 4 | 본문 텍스트 |
| `table` | 3 | 표 |
| `image` | 1 | 이미지/그림 |
| `equation` | 0 | 수식 |

---

## Tech Stack

### Frameworks & Libraries

| 도구 | 버전 | 용도 |
|------|------|------|
| **PyTorch** | 2.5.1+cu121 | 딥러닝 프레임워크 |
| **Ultralytics (YOLOv12)** | 8.3.185 | 문서 레이아웃 탐지 |
| **EasyOCR** | 1.7.2 | 한/영 다국어 텍스트 인식 |
| **CRAFT** | - | 문자 영역 탐지 (텍스트 디텍션) |
| **scikit-learn** | 1.7.1 | KMeans 멀티컬럼 감지 |
| **pdf2image** | 1.17.0 | PDF -> 이미지 변환 |
| **Pillow** | 11.3.0 | 이미지 처리 및 시각화 |
| **pandas** | 2.3.2 | CSV 데이터 핸들링 |
| **NumPy** | 2.2.6 | 수치 연산 |
| **Roboflow** | - | 데이터셋 어노테이션 |
| **Optuna** | - | 신뢰도 임계값 자동 최적화 |

### Training Environment

| 항목 | 사양 |
|------|------|
| GPU | NVIDIA HGX H200 |
| CUDA | 12.1 |
| 학습 시간 | ~4시간 (61 epochs, Early Stopped) |

---

## Training Results

### Fine-tuning Performance

| Epoch | Box Loss | Cls Loss | mAP50 | mAP50-95 | Precision | Recall |
|-------|----------|----------|-------|----------|-----------|--------|
| 1 | 1.128 | 3.720 | 0.132 | 0.100 | 0.086 | 0.453 |
| 10 | 0.595 | 0.664 | 0.789 | 0.627 | 0.735 | 0.724 |
| 30 | 0.468 | 0.362 | 0.849 | 0.682 | 0.806 | 0.818 |
| **61** | **0.422** | **0.292** | **0.852** | **0.701** | **0.841** | **0.809** |

### Module-wise Performance Improvement

```
                        Public Score
Baseline               ||||                          0.1168
+ Rule-based PP        ||||||                        0.245   (+110%)
+ Fine-tuning          ||||||||||||                  0.436   (+78%)
+ Optuna Optimization  |||||||||||||                 0.443   (+2%)
+ Reading Order        ||||||||||||||||||            0.457   (+3%)
+ OCR (800 DPI)        ||||||||||||||||||            0.4577  (final)
-----------------------------------------------------------------
Total Improvement       Baseline -> Final: +292%
```

---

## Quick Start

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

### 2. Training (Fine-tuning)

```bash
# 프로젝트 루트에서 실행
python scripts/train.py
```

주요 학습 설정은 `scripts/train.py` 상단의 CONFIG 블록에서 수정할 수 있습니다.

### 3. Inference

```bash
python scripts/inference.py
```

결과는 `outputs/submission.csv`로 저장됩니다.

---

## Project Structure

```
.
├── configs/                    # 설정 파일
│   ├── data.yaml               #   YOLO 데이터셋 설정
│   ├── dataset_info.txt        #   데이터셋 설명
│   └── roboflow_info.txt       #   Roboflow 어노테이션 정보
│
├── data/                       # 데이터셋 (자체 구축)
│   ├── raw/                    #   원본 데이터 (100 포스터 이미지 + YOLO 라벨)
│   │   ├── images/
│   │   └── labels/
│   ├── visualized/             #   BBox 시각화 샘플
│   ├── dataset_metadata.csv    #   데이터셋 메타 정보
│   └── data_license.csv        #   데이터 라이선스
│
├── models/                     # 모델 가중치
│   ├── pretrained/             #   사전학습 모델
│   │   ├── yolov12l-doclaynet.pt   # YOLOv12-L (DocLayNet)
│   │   ├── craft_mlt_25k.pth      # CRAFT 텍스트 탐지
│   │   ├── english_g2.pth         # EasyOCR 영어 모델
│   │   └── korean_g2.pth          # EasyOCR 한국어 모델
│   └── finetuned/              #   파인튜닝 모델
│       ├── V5.pt                   # 최종 제출 모델
│       ├── best.pt                 # 최고 성능 체크포인트
│       └── last.pt                 # 마지막 에폭 체크포인트
│
├── scripts/                    # 실행 스크립트
│   ├── train.py                #   학습 코드 (YOLOv12 Fine-tuning)
│   ├── inference.py            #   추론 코드 (Detection + OCR + Reading Order)
│   └── draw_pipeline.py        #   파이프라인 시각화
│
├── experiments/                # 학습 실험 로그 및 메트릭
│   ├── train_6cls_1024/        #   메인 실험 (imgsz=1024, 6 classes)
│   └── train_6cls_1024_v2/     #   2차 실험
│
├── docs/                       # 문서
│   ├── paper_draft.md          #   기술 논문 초안
│   ├── paper_summary.md        #   관련 연구 정리
│   ├── overleaf_paper/         #   LaTeX 논문 소스
│   ├── environment.txt         #   실행 환경 정보 (conda + GPU)
│   └── solution_report.pdf     #   솔루션 보고서
│
├── requirements.txt            # Python 의존성
└── .gitignore
```

---

## Model Licenses

| Model | License | Source |
|-------|---------|--------|
| YOLOv12 (Ultralytics) | AGPL-3.0 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) |
| YOLOv12-DocLayNet | AGPL-3.0 | [hantian/yolo-doclaynet](https://huggingface.co/hantian/yolo-doclaynet) |
| CRAFT (Text Detection) | Apache-2.0 | [clovaai/CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch) |
| EasyOCR (Korean/English) | Apache-2.0 | [JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR) |

---

## References

1. Ultralytics. *YOLOv12: Real-time Object Detection.* AGPL-3.0.
2. B. Pfitzmann et al. *DocLayNet: A Large Human-Annotated Dataset for Document-Layout Segmentation.* KDD, 2022.
3. Y. Baek et al. *Character Region Awareness for Text Detection.* CVPR, 2019.
4. JaidedAI. *EasyOCR: Ready-to-use OCR with 80+ Supported Languages.* Apache-2.0.
5. T. Akiba et al. *Optuna: A Next-generation Hyperparameter Optimization Framework.* KDD, 2019.
