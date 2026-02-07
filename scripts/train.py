#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Ultralytics YOLO 🚀, AGPL-3.0 license
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
# train.py — YOLO(ultralytics) 파인튜닝 스크립트
# - 모든 튜닝 변수를 맨 위 CONFIG 블록에서 한 번에 수정
# - 본문은 변경 없이 실행만 하면 됨
# - 실행:  python scripts/train.py  (프로젝트 루트에서)
#   (Ultralytics가 학습 진행 상황을 터미널에 자동 출력합니다)
# ============================================================

# ==============================
# CONFIG — 여기 값을 바꿔서 튜닝하세요
# ==============================

from pathlib import Path as _Path
PROJECT_ROOT = _Path(__file__).resolve().parent.parent

# [데이터 분할 관련]
SPLIT = True                 # True면 학습 전에 data/raw → train/valid/test로 분할을 수행
SPLIT_ROOT = str(PROJECT_ROOT / "data")        # 데이터 루트 폴더 (여기 하위에 raw/, train/, valid/, test/가 위치)
SPLIT_SRC = "raw"                              # 원본 데이터 폴더명 (SPLIT_ROOT/raw/에 images, labels가 있어야 함)
SPLIT_RATIOS = (0.9, 0.10, 0.00)  # (train, valid, test) 비율. 합이 1.0 이어야 함
SPLIT_MOVE = False            # True: 이동(move) → raw에서 파일이 사라짐 / False: 복사(copy)

WEIGHTS = str(PROJECT_ROOT / "models/pretrained/yolov12l-doclaynet.pt")  # 시작 가중치(사전학습)
DATA_YAML = str(PROJECT_ROOT / "configs/data.yaml")       # YOLO 데이터 설정 파일 경로
RUN_NAME = "train_6cls_1024"                               # 실험 이름. 결과는 experiments/<RUN_NAME>/에 저장됨

IMGSZ = 1024                  # 입력 이미지 한 변 크기. 정수면 레터박스로 정사각(1024x1024)로 들어감
BATCH = 16                     # 배치 크기. VRAM(예: RTX 4060 8GB)에 맞춰 조정
EPOCHS = 300                  # 학습 에폭 수
DEVICE = None                 # "0"(GPU0), "0,1"(멀티), "cpu". None이t면 자동 판별
WORKERS = 10                  # DataLoader 워커 수. I/O 병렬 로딩 스레드 개수

# [증강 관련] — 문서/레이아웃 도메인에서는 과도한 증강 지양
MOSAIC = 0.2                  # 모자이크 증강 확률(0~1). 문서 도메인에선 0.0~0.2 권장
CLOSE_MOSAIC = 10             # 마지막 N 에폭 동안 모자이크 비활성화
FLIPLR = 0.0                  # 좌우 반전 확률(0~1). 텍스트 왜곡 방지 위해 보통 0.0
TRANSLATE = 0.05              # 평행이동 범위(비율). 0.05 → ±5%
SCALE = 0.20                  # 스케일 변환 범위(비율). 0.20 → ±20%
HSV_H = 0.0                   # 색상(Hue) 변화 강도
HSV_S = 0.30                  # 채도(Saturation) 변화 강도
HSV_V = 0.20                  # 명도(Value) 변화 강도

# [학습 스케줄/조기종료/재현성]
COS_LR = True                 # 코사인 러닝레이트 스케줄 사용 여부
PATIENCE = 50                 # 조기 종료(EarlyStopping) 인내 에폭 수
SEED = 42                     # 난수 시드 (데이터 셔플/증강 재현)
DETERMINISTIC = True          # True면 재현성 우선(cuDNN 결정적) / False면 속도 우선

# ==============================
# CODE — 아래는 변경하지 않아도 됩니다
# ==============================
import random, shutil
from pathlib import Path

def _seed_all(seed=42, deterministic=True):
    """파이썬/넘파이/파이토치 난수 시드 고정(가능한 경우)"""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def split_from_train(root: Path, src="original", ratios=(0.8,0.1,0.1), move=False):
    """
    data/original/images & labels → train/valid/test 로 분할
    - root/src/images, root/src/labels에서 파일을 읽고
    - root/train|valid|test/(images|labels)로 복사/이동합니다.
    """
    root = Path(root)
    src_img = root/src/"images"
    src_lbl = root/src/"labels"
    if not (src_img.exists() and src_lbl.exists()):
        raise FileNotFoundError(f"원본 폴더가 없습니다: {src_img} / {src_lbl}")

    # 목적지 폴더 생성
    for s in ("train","valid","test"):
        (root/s/"images").mkdir(parents=True, exist_ok=True)
        (root/s/"labels").mkdir(parents=True, exist_ok=True)

    # 이미지 수집
    exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
    imgs = [p for p in src_img.iterdir() if p.suffix.lower() in exts]
    imgs.sort()
    random.shuffle(imgs)

    n = len(imgs)
    n_tr = int(n*ratios[0])
    n_va = int(n*ratios[1])
    splits = {
        "train": imgs[:n_tr],
        "valid": imgs[n_tr:n_tr+n_va],
        "test":  imgs[n_tr+n_va:]
    }

    def pair_label(ip: Path):
        """이미지와 같은 파일명(.txt) 라벨 경로 반환(없으면 None)"""
        q = src_lbl / ip.with_suffix(".txt").name
        return q if q.exists() else None

    moved = {k:0 for k in splits}
    for split, files in splits.items():
        for img in files:
            lbl = pair_label(img)
            di = root/split/"images"/img.name
            if move: shutil.move(str(img), str(di))
            else:    shutil.copy2(str(img), str(di))
            if lbl:
                dl = root/split/"labels"/lbl.name
                if move: shutil.move(str(lbl), str(dl))
                else:    shutil.copy2(str(lbl), str(dl))
            moved[split]+=1
    print("✅ split done:", moved)

def train():
    """Ultralytics YOLO 학습 실행. 터미널에 학습 진행 상황이 자동 출력됩니다."""
    _seed_all(SEED, DETERMINISTIC)
    try:
        from ultralytics import YOLO
    except Exception:
        raise RuntimeError("Ultralytics가 필요합니다. pip install ultralytics")

    model = YOLO(WEIGHTS)
    args = dict(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        mosaic=MOSAIC, close_mosaic=CLOSE_MOSAIC, fliplr=FLIPLR,
        translate=TRANSLATE, scale=SCALE,
        hsv_h=HSV_H, hsv_s=HSV_S, hsv_v=HSV_V,
        cos_lr=COS_LR, patience=PATIENCE,
        deterministic=DETERMINISTIC, seed=SEED,
        project=str(PROJECT_ROOT / "experiments"), name=RUN_NAME
    )
    print("[train args]", args)
    res = model.train(**args)

    # best.pt 백업
    save_dir = Path(getattr(res, "save_dir", PROJECT_ROOT/"experiments"/RUN_NAME))
    best = save_dir/"weights"/"best.pt"
    if best.exists():
        out = PROJECT_ROOT/"models"/"finetuned"/f"{RUN_NAME}_best.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, out)
        print("✅ best.pt copied →", out)
    else:
        print("⚠️ best.pt not found at", best)

if __name__ == "__main__":
    # 1) 선택적 데이터 분할
    if SPLIT:
        if abs(sum(SPLIT_RATIOS)-1.0) > 1e-6:
            raise ValueError("SPLIT_RATIOS의 합은 1.0이어야 합니다.")
        _seed_all(SEED, True)
        print("[split]", dict(root=SPLIT_ROOT, src=SPLIT_SRC, ratios=SPLIT_RATIOS, move=SPLIT_MOVE))
        split_from_train(Path(SPLIT_ROOT), SPLIT_SRC, SPLIT_RATIOS, SPLIT_MOVE)

    # 2) 학습 시작 (터미널 로그 자동 출력)
    print("[train start]")
    train()
