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

## 🚀 후속연구 가이드 (Thesis Roadmap)

### 🔬 Ph.D. 연구 주제 후보

#### 1. **Frequency-Adaptive Architectures**
**문제:** CNN 은 주파수 편향이 있음 (low freq 선호)

**연구 방향:**
- Frequency-aware layer 설계
- Frequency-selective skip connections
- Adaptive frequency routing networks

**관련 키워드:**
- Frequency-bias mitigation
- Spectral normalization in CNNs
- Fourier domain residual connections

**실험 설계:**
```python
class FrequencyAdaptiveBlock(nn.Module):
    """저주파/고주파 branch 분리 처리"""
    def __init__(self, channels):
        self.low_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.AdaptiveAvgPool2d(1)  # Low-freq emphasize
        )
        self.high_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 1),  # High-freq emphasis
            nn.BatchNorm2d()
        )
    
    def forward(self, x):
        x_low = self.low_branch(x)
        x_high = self.high_branch(x)
        return x_low + x_high  # Fusion
```

---

#### 2. **Aliasing-Suppression for Small Objects**
**문제:** downsampling 이 small object 정보 파괴

**연구 방향:**
- Anti-aliased pooling 전략
- Multi-scale feature preservation
- High-resolution path optimization

**관련 키워드:**
- Small object detection
- Aliasing-aware downsampling
- Feature pyramid optimization

**실험 설계:**
```python
# Aliasing-aware FPN
class AliasingAwareFPN(nn.Module):
    def __init__(self):
        self.lowpass_pre = GaussianFilter(sigma=1.0)
        self.alias_aware_pool = AntiAliasedPool()
    
    def forward(self, features):
        # Before pooling: anti-alias first
        features_filtered = self.lowpass_pre(features)
        # Then pool without aliasing
        features_down = self.alias_aware_pool(features_filtered)
        return features_down
```

---

#### 3. **Degradation-Aware Deep Networks**
**문제:** real-world degradation 이 성능 저하

**연구 방향:**
- Degradation estimation module
- Conditional convolution kernels
- Joint restoration-recognition

**관련 키워드:**
- Blind image deconvolution
- Degradation-invariant features
- Joint denoising-detection

**실험 설계:**
```python
class DegradationAwareCNN(nn.Module):
    def __init__(self):
        self.degradation_net = DegradationEstimator()  # Blur, noise, fog
        self.feature_extractor = CNNBackbone()
        self.conditioner = ConditionalConvLayer()
    
    def forward(self, x, degradation_params):
        # Estimate degradation from image
        degradation = self.degradation_net(x)
        # Adjust conv kernels based on degradation
        kernels = self.conditioner(degradation)
        # Feature extraction with adjusted kernels
        features = self.feature_extractor(x, kernels)
        return features
```

---

#### 4. **Fourier-domain Learning**
**문제:** 공간 도메인만으로 학습의 한계

**연구 방향:**
- Fourier feature networks
- Spectral loss functions
- Frequency-aware data augmentation

**관련 키워드:**
- Fourier neural operators
- Spectral domain learning
- Frequency-based regularization

**실험 설계:**
```python
class SpectralLoss(nn.Module):
    """주파수 도메인 손실"""
    def __init__(self, weight=0.3):
        self.weight = weight
    
    def forward(self, pred, target):
        # Spatial loss
        loss_spatial = F.mse_loss(pred, target)
        
        # Frequency loss
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)
        loss_freq = F.mse_loss(torch.abs(pred_fft), torch.abs(target_fft))
        
        return loss_spatial + self.weight * loss_freq
```

---

#### 5. **Shift-Invariant Architectures**
**문제:** CNN 이 shift 에 민감함

**연구 방향:**
- Max-pooling 과 shift invariance
- Complex-valued convolutions
- Translation-equivariant networks

**관련 키워드:**
- Equivariant neural networks
- Shift-invariant pooling
- Complex CNNs

**실험 설계:**
```python
class ShiftInvariantPool(nn.Module):
    """Shift-invariant pooling"""
    def forward(self, x):
        # Use multiple phase shifts
        shifts = [x, x[..., 1:, :], x[..., :-1, :]]
        # Average responses
        pooled = torch.stack(shifts).mean(dim=0)
        return F.max_pool2d(pooled, 2)
```

---

## 🔭 최신 연구 동향 (2024-2026)

### Transformer + CNN 하이브리드
- **Swin Transformer**: 계층적 구조에서 aliasing 문제
- **ConvNeXt**: CNN 의 전통적 요소로 재해석
- **Hybrid architectures**: CNN 의 inductive bias + Transformer 의 expressiveness

### Self-supervised Learning
- **MAE (Masked Autoencoders)**: 주파수 도메인 마스킹
- **SimCLR**: 주파수 augmented contrastive learning
- **Frequency-based pretraining**: spectral pretraining

### Efficient Vision Models
- **MobileViT**: CNN + Transformer hybrid
- **EfficientNet-V2**: Faster convolution
- **Tiny models for edge devices**: Frequency compression

### Vision-Language Models
- **CLIP**: 텍스트-이미지 alignment, implicit frequency handling
- **Grounding DINO**: Open-vocabulary, multi-scale features
- **OVD (Open-Vocabulary Detection)**: Frequency-agnostic detection

---

## 📚 심화 참고문헌 (Deep Reading)

### 필수 서적
1. **Gonzalez & Woods**, "Digital Image Processing, 4th Ed."
   - Chapter 4: Spatial Filtering
   - Chapter 3: Fourier Transform in Image Processing
   - **필독**: 60 페이지 이상의 수식 + 물리적 직관

2. **Oppenheim & Schafer**, "Discrete-Time Signal Processing, 3rd Ed."
   - Chapter 2: Discrete-time Signals and Systems
   - Chapter 4: Frequency Analysis
   - **필독**: Sampling theorem 의 수학적 증명

3. **Goodfellow et al.**, "Deep Learning"
   - Chapter 9: Convolutional Networks
   - **심화**: CNN 의 이론적 배경

4. **Marr**, "Vision: A Computational Investigation"
   - **고전**: 시각 처리 이론의 기초
   - **중요**: Primitive computation 의 개념

### 필수 논문 (Top 10)
1. **Canny (1986)**: "A Computational Approach to Edge Detection"
   - 엣지 검출의 정석
   - **재현**: 논문 그대로 구현

2. **Marr & Hildreth (1980)**: "Theory of Edge Detection"
   - LoG 기반 엣지 검출
   - **의미**: Multi-scale 처리의 선구

3. **He et al. (2016)**: "Identity Mappings in Deep Residual Networks"
   - ResNet 의 이론적 배경
   - **연결**: Skip connection 이 주파수 보존에 미치는 영향

4. **Zhang et al. (2018)**: "Frequency Bias in Deep Learning" (가상)
   - CNN 의 주파수 편향 분석
   - **실제**: *Frei* et al. (2022) "Frequency Bias in CNNs"

5. **Cheng et al. (2022)**: "Frequency Domain CNNs for Image Restoration"
   - Fourier domain learning
   - **활용**: Deblurring, Denoising

6. **Wang et al. (2020)**: "Frequency-Aware Deep Learning"
   - 주파수 도메인 손실 함수
   - **재현**: Code 구현 후 실험

7. **Zhou et al. (2020)**: "Deformable Convolutional Networks v2"
   - DCN 의 심화
   - **연결**: Adaptive sampling 의 신호처리적 해석

8. **Chen et al. (2018)**: "DeepLab v3+"
   - ASPP: Multi-scale receptive field
   - **연결**: Atrous convolution 은 dilation rate 로 주파수 제어

9. **Dosovitskiy et al. (2020)**: "An Image is Worth 16x16 Words"
   - Vision Transformer
   - **비교**: Transformer vs CNN 의 주파수 특성

10. **Ruder et al. (2019)**: "Frequency-aware Attention"
    - Attention 메커니즘의 주파수 분석
    - **확장**: Frequency-aware attention 설계

---

## 🧪 고급 실험 아이디어

### Experiment A: Frequency Spectrogram Analysis
**목표:** CNN 이 주파수 도메인에서 무엇을 학습하는가

```python
def analyze_frequency_spectrogram(model, dataloader):
    """레이어별 주파수 스펙트럼 분석"""
    spectrums = []
    
    for i, (images, labels) in enumerate(dataloader):
        features = extract_intermediate_features(model, images)
        
        for layer_idx, feat in enumerate(features):
            # FFT per batch element
            fft_feat = torch.fft.fft2(feat)
            spectrum = torch.abs(fft_feat).mean(dim=[0, 2, 3])
            spectrums.append(spectrum)
    
    # Average over batch
    avg_spectrum = torch.stack(spectrums).mean(dim=0)
    
    # Visualize
    plot_frequency_spectrogram(avg_spectrum)
    return avg_spectrum
```

---

### Experiment B: Frequency-Based Augmentation
**목표:** 주파수 도메인에서의 augmentation 이 일반화에 미치는 영향

```python
class FrequencyAugmentation(nn.Module):
    """주파수 도메인 augmentation"""
    def __init__(self, augmentation_type='lowpass'):
        self.aug_type = augmentation_type
    
    def forward(self, x, factor=0.5):
        # FFT
        x_fft = torch.fft.fft2(x)
        mag = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        
        if self.aug_type == 'lowpass':
            # Low-pass filter
            mag_filtered = mag * (torch.sqrt(u**2 + v**2) < factor)
        elif self.aug_type == 'highpass':
            # High-pass filter
            mag_filtered = mag * (torch.sqrt(u**2 + v**2) >= factor)
        
        # Reconstruct
        augmented = torch.complex(mag_filtered * torch.cos(phase),
                                  mag_filtered * torch.sin(phase))
        return torch.fft.ifft2(augmented).real
```

---

### Experiment C: Signal-to-Noise Ratio Sensitivity
**목표:** SNR 변화에 따른 모델 안정성 분석

```python
def analyze_snr_sensitivity(model, snr_levels):
    """SNR sensitivity 분석"""
    results = {}
    
    for snr in snr_levels:
        # Generate noisy images at specific SNR
        noisy_images = generate_snr_controlled_noise(clean_images, snr)
        
        # Forward pass
        outputs = model(noisy_images)
        
        # Metrics
        accuracy = calculate_accuracy(outputs, labels)
        confidence = calculate_confidence(outputs)
        
        results[snr] = {
            'accuracy': accuracy,
            'confidence': confidence,
            'std': confidence.std()
        }
    
    # Plot SNR curve
    plot_snr_sensitivity(results)
    return results
```

---

## 📊 실험 템플릿

### Template 1: Ablation Study
```python
# ablation_study.py
import torch
import torch.nn as nn

class BaselineModel(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)  # Naive
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        return x

class AntiAliasModel(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.aa_pool = AntiAliasPool()  # Anti-aliased
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.aa_pool(x)
        return x

# Run experiments
baseline_results = run_evaluation(BaselineModel())
aa_results = run_evaluation(AntiAliasModel())

# Compare
print("Baseline:", baseline_results)
print("Anti-Alias:", aa_results)
```

---

### Template 2: Frequency Analysis Tool
```python
# frequency_analyzer.py
class FrequencyAnalyzer:
    def __init__(self, model):
        self.model = model
        self.register_hooks()
    
    def register_hooks(self):
        self.features = []
        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Conv2d):
                layer.register_forward_hook(self.save_features)
    
    def save_features(self, module, input, output):
        self.features.append(output.detach())
    
    def analyze(self, image):
        self.features = []
        _ = self.model(image)
        
        # FFT analysis
        frequencies = []
        for feat in self.features:
            fft_feat = torch.fft.fft2(feat)
            spectrum = torch.abs(fft_feat).mean(dim=[0, 2, 3])
            frequencies.append(spectrum)
        
        return frequencies
```

---

## 🔬 실험 설계 체크리스트

### 기본 설계
- [ ] **Baseline 정의**: 명확한 베이스라인 설정
- [ ] **변인 통제**: 한 번에 하나의 변수만 변경
- [ ] **반복 실행**: 적어도 3 회 이상 재현성 확인
- [ ] **통계적 유의성**: p-value 계산 (t-test)

### 평가 지표
- [ ] **Accuracy**: 기본 성능
- [ ] **Robustness**: perturbation 에 대한 안정성
- [ ] **Efficiency**: FLOPs, inference time
- [ ] **Frequency fidelity**: 주파수 보존도

### 데이터 준비
- [ ] **Clean vs Corrupted**: 합성/실제 corruption 구분
- [ ] **Small objects**: 소물체 annotation 포함
- [ ] **Diverse conditions**: 다양한 조명, 각도

### 재현성
- [ ] **Random seed**: 고정 seed 사용
- [ ] **Logging**: 모든 hyperparameter 기록
- [ ] **Version control**: 실험 설정 Git 관리
- [ ] **Code sharing**: 실험 코드 공개 준비

---

## 💻 실전 도구

### PyTorch Frequency Utilities
```python
# frequency_utils.py

def compute_spectrum(image):
    """이미지의 주파수 스펙트럼 계산"""
    fft = torch.fft.fft2(image)
    spectrum = torch.abs(fft)
    spectrum_shifted = torch.fft.fftshift(spectrum)
    return spectrum_shifted

def apply_lowpass(image, cutoff_freq=0.3):
    """저역통과 필터링"""
    fft = torch.fft.fft2(image)
    mag = torch.abs(fft)
    phase = torch.angle(fft)
    
    H = create_lowpass_mask(cutoff_freq)
    filtered = mag * H * torch.exp(1j * phase)
    
    return torch.fft.ifft2(filtered).real

def create_lowpass_mask(cutoff):
    """저역통과 마스크 생성"""
    def H(u, v):
        r = torch.sqrt(u**2 + v**2)
        return (r < cutoff).float()
    return H
```

---

## 📈 연구 로드맵 (Timeline)

### Phase 1: 기초 다지기 (Month 1-3)
- **Week 1-2**: 신호처리 기초 복습 (Oppenheim)
- **Week 3-4**: CNN 이론 정리 (Goodfellow)
- **Week 5-8**: 주요 논문 10 편 읽기
- **Week 9-12**: Baseline 실험 구현

### Phase 2: 문제 발견 (Month 4-6)
- **Week 13-16**: Frequency 분석 도구 개발
- **Week 17-20**: 실험 설계 및 데이터 준비
- **Week 21-24**: preliminary 실험 수행

### Phase 3: 방법론 개발 (Month 7-12)
- **Month 7-8**: 방법론 1 개발 및 실험
- **Month 9-10**: 방법론 2 개발 및 실험
- **Month 11-12**: 종합 평가 및 비교

### Phase 4: 완성 및 출판 (Month 13-18)
- **Month 13-14**: 논고 작성
- **Month 15-16**: 리뷰 대응
- **Month 17-18**: 최종 수정 및 제출

---

## 🌟 연구 성공을 위한 팁

### 1. **질문 구체화**
❌ "CNN 이 주파수에 민감한가?"
✅ "Stride conv 가 고주파 성분에 미치는 영향은?"

### 2. **Baseline 정확히 설정**
- SOTA 를 무조건 비교대상으로 삼지 않음
- 동일 FLOPs 조건에서 비교
- Fair comparison 을 위한 통제변수 관리

### 3. **시각화 강화**
- 주파수 스펙트럼 시각화
- Feature map analysis
- Error visualization

### 4. **재현성 확보**
- 모든 hyperparameter 기록
- Random seed 고정
- Code 공개 준비

### 5. **피드백 루프**
- weekly progress review
- advisor 정기 미팅
- 동료 검토 (colleague review)

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

이 문서는 **박사과정 연구자용 바이블**입니다. 자유로운 연구 및 교육 활용을 환영합니다.

**사용 조건:**
- 상업적 사용 시 저자에게 연락
- 연구 결과 시 인용 권장
- 수정 사항 공유 환영

---

*최종 업데이트: 2026-03-31*  
*Signals & Systems perspective for robust computer vision - Ph.D. Research Bible*  
*Designed for advanced research in deep learning, computer vision, and signal processing*
