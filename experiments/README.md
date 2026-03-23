# Experiments: Signal-Aware CV Ablations (Toy)

외부 데이터셋 없이 바로 실행 가능한 toy ablation 스크립트 세트입니다.

## 준비
```powershell
cd C:\Users\ust21\signals-and-cv
python -m pip install -r .\experiments\requirements.txt
```

## 실행
```powershell
python .\experiments\anti_aliasing_stride_ablation.py
python .\experiments\small_object_aliasing_ablation.py
python .\experiments\robust_perception_toy_ablation.py
```

## 출력
- 기본 출력 경로: `experiments/results/`
- 각 스크립트가 `*.json` 결과 파일 생성

## 스크립트 구성
- `anti_aliasing_stride_ablation.py`
  - naive stride downsample vs anti-aliased downsample
  - shift-consistency 및 high-frequency 보존 지표 비교
- `small_object_aliasing_ablation.py`
  - 소물체(1~2 px) 검출 score의 shift 민감도 비교
- `robust_perception_toy_ablation.py`
  - weather/noise/blur 열화에서 전처리 파이프라인별 edge/PSNR trade-off 비교
