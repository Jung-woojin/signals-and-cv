# Signals and Computer Vision 📊👁️

실환경 컴퓨터비전과 CNN 아키텍처 연구를 위한 **신호및시스템 (Signals & Systems) 관점** 문서 및 실험 리포지토리입니다.

> 🔥 **핵심 철학**: 이미지와 영상은 본질적으로 **이산 신호**입니다. CNN 은 단순한 신경망이 아니라 **학습형 신호처리 시스템**입니다.

---

## 📚 목차

### 📖 핵심 문서
- [신호및시스템이 왜 CNN 과 현대 컴퓨터비전에 중요한가](./docs/why-signals-systems-matter-for-cnn-and-cv.md)
- [공간주파수와 영상필터링](./docs/spatial-frequency-and-filters.md)

### 🧪 실험 (Experiments)
- [바로 실행 가능한 toy ablation 스크립트](./experiments/README.md)

---

## 🎯 핵심 메시지 5 가지

### 1️⃣ CNN 은 학습형 신호처리 시스템입니다
**이미지 = 이산 신호**, **CNN = 적응형 필터뱅크**

고전 신호처리의 필터링/convolution/frequency/sampling 이 CNN 의 학습 연산으로 확장된 형태입니다.

### 2️⃣ 신호관점으로 성능 저하 원인 분해
- **Aliasing** → Shift perturbation 에 민감
- **고주파 손실** → Small object, 미세 경계 취약
- **SNR 감소** → Robust perception 저하

### 3️⃣ 고전 영상처理的도 여전히 유효합니다
Blur, denoise, edge, deconvolution 은 CNN 의 전처리/손실/모듈설계에 직접 활용됩니다.

### 4️⃣ 실환경 문제는 신호 관점이 필수입니다
안개/비/저조도/원거리 소물체 → **SNR 저하, 고주파 손실, 샘플링 왜곡**

### 5️⃣ 좋은 아키텍처 = 주파수 대역 제어
단순히 깊은 네트워크가 아니라 **주파수 대역 제어·샘플링 안정성·복원 가능성**을 함께 설계합니다.

---

## 🔬 주요 개념

### 🎨 주파수 관점에서 본 이미지

**저주파 (Low Frequency)**
- 조명 변화, 큰 물체, 배경 구조
- 이미지의 "대략적인 형태"
- CNN 이 빠르게, 안정적으로 학습

**고주파 (High Frequency)**
- 경계선 (edge), 텍스처, 미세 디테일
- 이미지의 "세부 정보"
- CNN 이 느리게, 불안정하게 학습

```
주파수 스펙트럼:
  ┌─────────────────┐
  │   ···    ···    │  ← 고주파 영역 (외곽)
  │   ·  ···  ·  ·  │
  │ ·   ·   ·   ·  │
  │   ·   CORE   ·  │  ← 저주파 영역 (중앙)
  │ ·   ·   ·   ·  │
  │   ·  ···  ·  ·  │
  │   ···    ···    │
  └─────────────────┘
```

### 📊 필터링과 Convolution

**고전 필터링** ↔ **CNN 의 Convolution**

| 고전 | CNN |
|------|-----|
| 고정 필터 (Sobel, Gaussian) | 학습 가능한 커널 |
| 수동 설계 | 데이터 기반 최적화 |
| 단일 목적 | 다목적 특성 추출 |

```
고전:  y = x * h  (h: 고정)
CNN:   y = x * W  (W: 학습)
```

### 🚨 Aliasing 의 함정

**다운샘플링 전에 anti-aliasing 필터가 없으면:**
- 가짜 패턴 생성 (moiré)
- Shift perturbation 에 민감
- Small object 정보 손실

```python
# naive downsampling
y_naive = image[..., ::2, ::2]

# anti-aliased downsampling
k = torch.tensor([[1.,2.,1.],[2.,4.,2.],[1.,2.,1.]])
k = k / k.sum()  # Normalize
image_lp = F.conv2d(image, k, padding=1)  # Low-pass
y_aa = image_lp[..., ::2, ::2]
```

### 🌧️ Robust Perception

**실환경 열화 = 신호 관점 해석**

| 문제 | 신호 관점 | 해결 |
|------|------|-----|
| 저조도 | SNR 감소 | Denoise |
| 블러 | 고주파 손실 | Deblur |
| 안개 | Contrast 저하 | Dehaze |
| rain/snow | 노이즈 증가 | Denoise + Robust loss |

---

## 🛠️ 실험 (Experiments)

### 📦 준비

```powershell
cd experiments
pip install -r requirements.txt
```

### 🚀 실행

```powershell
# Anti-aliasing ablation
python anti_aliasing_stride_ablation.py

# Small object aliasing ablation
python small_object_aliasing_ablation.py

# Robust perception ablation
python robust_perception_toy_ablation.py

# Generate report
python report.py
```

### 📊 실험 내용

#### 1. Anti-aliasing Stride Ablation
**Naive downsampling vs Anti-aliased downsampling**

- Shift consistency 비교
- High-frequency 보존 지표 측정
- Clean accuracy vs Robustness trade-off 분석

#### 2. Small Object Aliasing Ablation
**1~2px 소물체 검출의 shift 민감도**

- Downsampling 정책별 소물체 detection score
- Aliasing 이 소물체 감지에 미치는 영향

#### 3. Robust Perception Ablation
**Weather/Noise/Blur 열화에서 전처리 비교**

- Edge preservation vs PSNR trade-off
- 전처리 파이프라인별 robustness 평가

---

## 🎨 필터링 종류

### 🔴 미분필터 (Differential Filters)

**Sobel Filter**

```
Gx (수평 엣지):  Gy (수직 엣지):
[-1  0  1]      [-1 -2 -1]
[-2  0  2]  vs   [ 0  0  0]
[-1  0  1]      [ 1  2  1]
```

- 노이즈 강건
- Edge magnitude: `√(Gx² + Gy²)`

**Laplacian Filter (2 차 미분)**

```
[ 0  -1   0]
[-1   4  -1]
[ 0  -1   0]
```

- 엣지 두께 검출
- 노이즈 민감
- Image sharpening 에 사용

### 🟢 저역필터 (Low-Pass Filters)

**Gaussian Filter**

```
[0.05  0.12  0.05]
[0.12  0.33  0.12]
[0.05  0.12  0.05]
```

- 부드러운 블러
- Rotation symmetric
- Anti-aliasing 에 필수

**Median Filter**
- 소금-후추 노이즈 제거
- 엣지 보존
- 비선형 필터

### 🔵 고역필터 (High-Pass Filters)

**Unsharp Masking**

```
1. Blur: B = Gaussian(I)
2. Mask: M = I - B
3. Sharpen: O = I + α·M
```

- 이미지 선명화
- 고주파 강조
- 자연스러운 효과

### 🟡 Anisotropic Filters

**Directional Sobel**

```
G(θ) = Gx·cosθ + Gy·sinθ
```

- 특정 방향 엣지 검출
- Texture analysis
- Orientation detection

---

## 🧠 아키텍처 설계 힌트

### 1. Anti-aliasing-aware Backbone
- Stride conv 전에 저역필터 삽입
- Shift robustness 비교
- Blur pooling 도입

### 2. Frequency-selective Feature Routing
- Low-frequency branch
- High-frequency branch
- Task-based fusion

### 3. Degradation-conditioned Convolution
- Blur/noise/weather 강도 추정
- 동적 커널 조정
- 조건부 convolution

### 4. Fourier-domain Auxiliary Loss
- 공간 도메인 loss + 주파수 loss
- 주파수 재구성 정확도 최적화
- High-frequency consistency

### 5. Small-object Preservation Block
- Early downsampling 억제
- High-frequency path 강화
- Aliasing suppression 결합

---

## 📈 실험 설계 체크리스트

- [ ] 동일 FLOPs 조건에서 anti-aliasing 유무 비교
- [ ] mAP/IoU 외에 shift consistency, weather robustness 지표 추가
- [ ] Synthetic degradation vs Real-world degradation 분리 평가
- [ ] 입력 파이프라인 (보간법/압축률) 민감도 분석
- [ ] Feature map FFT magnitude 레이어별로 기록

---

## 🔍 탐색 키워드

- **Anti-aliased CNN**
- **Frequency bias in deep networks**
- **Spectral regularization for vision**
- **Fourier feature networks**
- **Degradation-aware perception**
- **Blur-robust detection**
- **Weather-robust visual recognition**
- **Small object detection under low SNR**
- **Joint restoration and recognition**
- **Shift-invariant architectures**

---

## 📚 참고 자료

### 서적
1. **Gonzalez & Woods**, "Digital Image Processing" - 기본
2. **Marr**, "Vision: A Computational Investigation" - 이론
3. **Deep Learning**, Goodfellow et al. - 딥러닝

### 논문
- **Canny**, "A Computational Approach to Edge Detection"
- **Marr & Hildreth**, "Theory of Edge Detection" (LoG)
- **Anti-aliased CNN** 관련 최신 논문들

---

## 📝 License

이 문서는 **연구 및 교육 목적**으로 제작되었습니다. 자유롭게 활용하세요.

---

*최종 업데이트: 2026-03-31*
*Signals & Systems perspective for robust computer vision*
