# Volume 2 — 디퓨전 모델의 기술적 이해

> **이 권의 목적:**
> "이미지 생성 AI가 어떻게 그림을 만드는지" 의 동작 원리를 **수학·코드 양쪽 레벨**에서 이해한다.
> DDPM부터 Rectified Flow까지의 **진화 흐름과 그 이유**를 빠짐없이 따라간다.
>
> **이 권을 다 읽으면 할 수 있는 것:**
> 1. 디퓨전 모델이 왜 GAN/VAE를 압도했는지 한 문단으로 설명할 수 있다.
> 2. DDPM이 1000 step에서 DDIM 50 step → Rectified Flow 4 step으로 줄어든 **각 단계의 트릭**을 설명할 수 있다.
> 3. Latent Diffusion이 Stable Diffusion 폭발의 결정적 발명인 이유를 안다.
> 4. CFG가 매 step 모델을 두 번 호출하는 이유와, FLUX.2 Klein이 CFG=1.0인 의미를 안다.
> 5. 디퓨전 추론 1 cycle의 코드를 손으로 따라 쓸 수 있다.
> 6. 왜 LLM의 KV cache가 디퓨전에는 직접 적용되지 않는지 설명할 수 있다.

> **분량:** 약 2,400줄 / 정독+실습 시 8~12시간.

---

## 목차

```
[0]   시작하기 전에 — 1권에서 가져온 것, 이 권에서 추가할 것
[1장] 생성 모델의 짧은 역사 — 왜 디퓨전이 살아남았나
[2장] DDPM — 디퓨전의 원조, 두 방향의 마르코프 사슬
[3장] DDPM 수학 친화 풀이 — 백엔드 개발자가 이해하는 forward/reverse
[4장] DDIM — 1000 step을 50 step으로 줄인 결정적 트릭
[5장] Latent Diffusion — Stable Diffusion이 폭발한 진짜 이유
[6장] Rectified Flow — 곡선 경로를 직선으로 펴기
[7장] Distillation — Teacher의 50 step을 Student의 4 step으로
[8장] Classifier-Free Guidance (CFG) — 매 step 두 번 도는 이유
[9장] 디퓨전 추론 한 cycle 코드로 분해하기
[10장] 디퓨전 vs LLM — KV cache가 안 되는 이유와 그 함의
[11장] 실습: Stable Diffusion 1.5 로컬 추론 + 시간 측정
[부록 A] 자주 헷갈리는 수식 사전
[부록 B] 디퓨전 모델 패밀리 트리
```

---

# 0장. 시작하기 전에

## 🎯 목적

1권에서 익힌 PyTorch와 텐서 직관을 **이 권의 디퓨전 모델 분해**에 어떻게 연결할지 명시한다.

## 0.1 1권에서 가져오는 것

이 권은 다음을 전제한다.

| 1권 챕터 | 이 권에서 활용 |
|---|---|
| 4장 (행렬·텐서) | 디퓨전 latent의 shape 읽기 |
| 5장 (신경망) | UNet/DiT의 forward 흐름 |
| 6장 (PyTorch) | nn.Module / `.eval()` / `no_grad()` |
| 7장 (HF/SD 추론) | 이번 권 11장 실습의 출발점 |

만약 위 항목 중 어느 하나라도 흐릿하다면 1권의 해당 장을 다시 보고 오라.

## 0.2 이 권에서 새로 들이는 어휘

| 새 어휘 | 한 줄 정의 |
|---|---|
| **Forward process** | 깨끗한 이미지에 노이즈를 점진적으로 더하는 과정 |
| **Reverse process** | 노이즈에서 깨끗한 이미지를 복원하는 과정 (실제 모델) |
| **Noise schedule** | 각 step에서 얼마나 노이즈를 더할지의 표 |
| **Variance preserving** | DDPM의 잡음 누적 방식 |
| **Score function** | 데이터 분포의 그래디언트 |
| **Latent space** | VAE로 압축된 저차원 공간 |
| **Sampler** | reverse 과정을 실제로 풀어내는 알고리즘 |
| **Velocity** | Rectified Flow가 학습하는 양 |
| **Step (denoising step)** | reverse 과정의 한 번의 노이즈 제거 |

## 0.3 백엔드 친화 마인드셋 — 이 권에서 사용할 비유 풀

| 디퓨전 개념 | 백엔드 비유 |
|---|---|
| Forward process | DB에 의도적으로 noise를 끼워넣는 시뮬레이션 |
| Reverse process | 망가진 데이터에서 원본 복원하는 ETL |
| Noise schedule | TTL/스케줄링 정책 |
| Latent space | 압축 / Redis serialization 형식 |
| Sampler | DB 마이그레이션 정책 (단순 vs 영리) |
| Distillation | 큰 ETL 잡을 요약 ETL로 압축 |

이 비유들은 본문에서 반복적으로 등장한다.

---

# 1장. 생성 모델의 짧은 역사

## 🎯 이 장의 목적

**왜 지금 디퓨전인가**의 답을 역사적 흐름으로 안다. GAN, VAE, Diffusion의 본질적 차이와 트레이드오프를 정리한다.

## 1.1 개요

이미지 생성 모델은 2014년부터 다음 흐름으로 발전했다.

```
2014  GAN (Generative Adversarial Network)
2014  VAE (Variational AutoEncoder)
2015  PixelRNN/PixelCNN (autoregressive image)
2020  DDPM (Denoising Diffusion)             ← 디퓨전의 원조
2021  DDIM (가속화)
2022  Latent Diffusion / Stable Diffusion    ← 폭발의 시작
2022  Imagen, DALL-E 2 (텍스트→이미지)
2023  SDXL, Diffusion Transformer (DiT)
2024  Stable Diffusion 3, FLUX.1            ← 현재 SOTA 진입
2024  Rectified Flow + Distillation (4 step) ← 이 챌린지의 모델 계열
```

각 단계마다 **무엇이 풀리고 무엇이 새로 떠올랐는지**를 본다.

## 1.2 GAN — 2014년의 혁명, 그러나 한계

### 1.2.1 핵심 아이디어

**Generator** 와 **Discriminator** 두 네트워크가 적대적으로 학습한다.

```
Generator   : 가짜 이미지를 만든다 (위조범)
Discriminator: 진짜 vs 가짜를 구분한다 (감별사)

→ 둘이 서로를 이기려고 학습하면서 점점 좋아짐
```

### 1.2.2 장점

- **추론이 매우 빠름**: 한 번의 forward만 하면 이미지 완성.
- **고화질**: StyleGAN 시리즈는 인간 얼굴을 매우 사실적으로 만듦.

### 1.2.3 한계 (왜 죽어가는가)

1. **Mode collapse**: Generator가 몇 가지 패턴만 반복 생성. 다양성 부족.
2. **학습 불안정**: Generator/Discriminator의 균형 잡기 까다로움.
3. **텍스트 조건부 생성에 약함**: "강아지가 모자 쓴 그림" 같은 자유로운 텍스트로 잘 안 됨.
4. **데이터셋 한정**: 얼굴, 동물 등 좁은 도메인에 강하고 일반화 약함.

> **핵심**: 빠르지만 **다양성·조건부 생성·확장성**에서 디퓨전에 뒤졌다.

## 1.3 VAE — 안정적 학습, 그러나 흐릿함

### 1.3.1 핵심 아이디어

**Encoder** 와 **Decoder** 가 데이터를 latent space로 압축·복원한다.

```
이미지 ──Encoder──→ 잠재 분포 (보통 가우시안) ──샘플링──→ Decoder──→ 이미지
```

### 1.3.2 장점

- **학습 안정**.
- **수학적으로 깨끗함** (variational inference).
- **latent space가 의미 있음** (이미지 편집, 보간 가능).

### 1.3.3 한계

1. **출력이 흐릿함**: 평균을 학습하는 경향 때문에 디테일이 뭉개짐.
2. **품질이 GAN보다 낮음**.

### 1.3.4 살아남은 이유

VAE는 단독으로는 거의 안 쓰이지만, **Latent Diffusion의 압축 단계**로 핵심 역할을 한다 (5장에서 다룸).

## 1.4 Diffusion — 두 길의 장점을 모두 흡수

### 1.4.1 핵심 아이디어

이미지에 **노이즈를 점진적으로 더했다가**, 그 역과정을 학습한다.

```
clean image ──+ noise── x_1 ──+ noise── x_2 ── ... ──+ noise── x_T (pure noise)
                                                                         │
   ←──── 모델이 이 역방향을 학습 (reverse process) ────────────────────┘
```

### 1.4.2 왜 강한가

| 측면 | GAN | VAE | Diffusion |
|---|---|---|---|
| 품질 | 높음 | 흐림 | **매우 높음** |
| 다양성 | 낮음 (mode collapse) | 보통 | **매우 높음** |
| 학습 안정 | 어려움 | 안정 | **안정** |
| 텍스트 조건부 | 약함 | 약함 | **강함** |
| 추론 속도 | **빠름 (1 step)** | 빠름 | 느림 (수십~수백 step) |

→ **딱 하나, 추론 속도만 약점**. 그래서 이 챌린지(`task-ai`)가 정확히 그 약점을 푸는 게임이다.

## 1.5 핵심 원리 — 디퓨전이 살아남은 결정적 이유 3가지

```
┌──────────────────────────────────────────────────────┐
│ 1. 점진적 학습 = 안정성                              │
│    한 번에 이미지를 만들지 않고 조금씩 변형 →        │
│    학습이 GAN처럼 폭발하지 않는다                    │
│                                                        │
│ 2. 분포 모델링 = 다양성                              │
│    데이터의 확률 분포를 통째로 모델링 →               │
│    mode collapse가 일어나지 않는다                    │
│                                                        │
│ 3. 조건부 주입이 자연스러움                           │
│    매 step의 입력에 텍스트 임베딩을 끼워넣으면 됨 →   │
│    프롬프트 따르기가 강력하다                         │
└──────────────────────────────────────────────────────┘
```

이 3가지 때문에 **디퓨전이 GAN을 밀어냈다**. 그러나 추론 속도라는 마지막 한계를 풀어가는 흐름이 이 권의 후반부 (DDIM → Distillation → Rectified Flow) 이다.

## 1.6 시각자료 — 패러다임 비교

```
   GAN                    VAE                     Diffusion
──────                  ──────                  ──────────

[noise]                 [image]                  [image]
   │                       │                        │
   ▼                       ▼                        ▼
[Generator]             [Encoder]            [+ small noise]
   │                       │                        │ (T번 반복)
   ▼                       ▼                        ▼
[image]                  [latent]            [pure noise]
   │                       │                        │
   ▼                       ▼                        │ ◄─ 이 역방향을
[Discriminator]          [Decoder]                  │    모델이 학습
   │                       │                        │
   ▼                       ▼                        ▼
[real vs fake]          [image]                 [image]
```

## 1.7 자가 점검

1. GAN의 한계 4가지를 들어라.
2. VAE의 출력이 흐릿한 이유는?
3. 디퓨전이 GAN을 밀어낸 결정적 이유 3가지는?
4. 디퓨전의 유일한 약점은 무엇이며, 이 챌린지가 그것과 어떻게 연관되는가?
5. VAE는 왜 죽지 않고 살아남았는가?

---

# 2장. DDPM — 디퓨전의 원조

## 🎯 이 장의 목적

**DDPM (Denoising Diffusion Probabilistic Models, 2020)** 의 두 가지 핵심 — forward / reverse process — 를 직관과 코드 양쪽으로 익힌다.

## 2.1 개요

DDPM(Ho et al., 2020)이 디퓨전 모델 시대를 열었다. 핵심 발상:

> **노이즈를 더하는 forward process** 와 **노이즈를 빼는 reverse process** 를 둘 다 마르코프 사슬로 정의하고, reverse를 신경망으로 학습한다.

## 2.2 기초 — Forward Process

### 2.2.1 정의

T개의 step을 거쳐 깨끗한 이미지 $x_0$ 에 노이즈를 점진적으로 더한다.

각 step에서:
$$
x_t = \sqrt{1-\beta_t} \cdot x_{t-1} + \sqrt{\beta_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

여기서 $\beta_t$ 는 step별 노이즈 강도. 작은 값(예: 0.0001 ~ 0.02).

### 2.2.2 백엔드 친화 풀이

위 수식을 코드로 보면 즉시 직관이 선다.

```python
import torch

def forward_step(x_prev, beta_t):
    noise = torch.randn_like(x_prev)
    x_t = torch.sqrt(1 - beta_t) * x_prev + torch.sqrt(beta_t) * noise
    return x_t

# 예: 이미지에 1000 step에 걸쳐 노이즈 더하기
x = clean_image  # shape (1, 3, 64, 64)
betas = torch.linspace(0.0001, 0.02, 1000)

for t in range(1000):
    x = forward_step(x, betas[t])

# x는 이제 거의 순수 가우시안 노이즈
```

### 2.2.3 핵심 트릭 — 한 번에 t step 점프

매 step을 하나씩 돌리지 않고, **임의의 t step에 한 번에 도달**할 수 있다.

$$
x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon
$$

여기서 $\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)$.

**왜 가능한가**: 가우시안 분포의 합은 가우시안이라는 성질.

```python
def forward_jump(x_0, t, alpha_bar):
    noise = torch.randn_like(x_0)
    x_t = torch.sqrt(alpha_bar[t]) * x_0 + torch.sqrt(1 - alpha_bar[t]) * noise
    return x_t, noise
```

학습 시 매번 1000 step 돌리는 게 아니라 **랜덤 t를 골라 한 번에 점프** 한다. 그래서 학습이 가능한 속도가 됐다.

## 2.3 핵심 원리 — Reverse Process

### 2.3.1 정의

Forward와 정확히 반대 방향으로 노이즈를 제거한다.

$$
x_{t-1} = \mu_\theta(x_t, t) + \sigma_t \cdot \epsilon
$$

여기서 $\mu_\theta$ 는 **신경망이 예측** 한다. $\sigma_t$ 는 정해진 분산.

### 2.3.2 신경망이 예측하는 것

DDPM 논문의 결정적 단순화:

> **모델은 $\mu_\theta$ 가 아니라 "지금 들어있는 노이즈 $\epsilon$"를 예측한다.**

즉:
$$
\epsilon_\theta(x_t, t) \approx \epsilon \quad \text{(실제로 더해진 노이즈)}
$$

### 2.3.3 학습 손실

매우 단순하다.

$$
\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$

코드로:

```python
def loss_ddpm(model, x_0, alpha_bar):
    batch_size = x_0.shape[0]
    t = torch.randint(0, 1000, (batch_size,))
    
    # 한 번에 t step 점프
    noise = torch.randn_like(x_0)
    x_t = torch.sqrt(alpha_bar[t]) * x_0 + torch.sqrt(1 - alpha_bar[t]) * noise
    
    # 모델이 노이즈 예측
    pred_noise = model(x_t, t)
    
    # MSE loss
    return ((noise - pred_noise) ** 2).mean()
```

이 한 함수가 **DDPM 학습의 전부**다.

### 2.3.4 추론 (Sampling)

학습이 끝났으면 노이즈에서 시작해 reverse를 한 step씩 풀어낸다.

```python
def sample_ddpm(model, shape, T, betas, alpha_bar):
    x = torch.randn(shape)  # pure noise
    
    for t in reversed(range(T)):  # T-1, T-2, ..., 1, 0
        # 모델이 노이즈 예측
        pred_noise = model(x, torch.tensor([t]))
        
        # 한 step 노이즈 제거
        alpha_t = 1 - betas[t]
        coef_x = 1 / torch.sqrt(alpha_t)
        coef_eps = (1 - alpha_t) / torch.sqrt(1 - alpha_bar[t])
        
        x = coef_x * (x - coef_eps * pred_noise)
        
        if t > 0:
            x = x + torch.sqrt(betas[t]) * torch.randn_like(x)
    
    return x  # 이제 깨끗한 이미지
```

이게 **DDPM 추론의 전부**다. 단, **T번 (보통 1000번) 반복**해야 한다는 게 치명적 단점.

## 2.4 자세한 내용 — 왜 1000 step인가

### 2.4.1 직관적 이유

각 step에서 더하는 노이즈가 충분히 작아야:
- Forward의 가우시안 가정이 잘 성립.
- Reverse를 학습하기 쉬워짐.

너무 큰 step을 쓰면 가우시안 분포로 근사가 안 된다.

### 2.4.2 실용적 비용

1000 step × 100ms/step = **100초/이미지**. 서비스에 못 씀.

→ DDIM (4장)이 이 문제를 해결한다.

## 2.5 시각자료 — DDPM 흐름

```
학습 시:
                  랜덤 t 선택
                      │
[x_0 (clean)] ──(forward jump)──→ [x_t]
                                    │
                                    ▼
                              [Model: ε_θ(x_t, t)]
                                    │
                                    ▼
                              [예측 노이즈]
                                    │ (실제 ε와 MSE)
                                    ▼
                                  [Loss]


추론 시:
[x_T (pure noise)]
        │
        ▼
   [model] → ε_θ → 한 step 제거 → [x_{T-1}]
        │
        ▼
   [model] → ε_θ → 한 step 제거 → [x_{T-2}]
        │
        ▼
        ...  (T번 반복, 1000번)
        │
        ▼
   [x_0 (clean image)]
```

## 2.6 자가 점검

1. Forward process는 무엇을 하는가? 한 식으로 적어라.
2. Forward의 한 번 점프 트릭은 왜 가능한가?
3. DDPM 모델이 예측하는 양은 무엇인가?
4. DDPM 학습 손실 함수를 한 줄 코드로 적어라.
5. DDPM 추론이 왜 느린가? 그 한계가 어떻게 풀려갔는가?

---

# 3장. DDPM 수학 친화 풀이

## 🎯 이 장의 목적

2장의 수식을 **백엔드 개발자가 익숙한 비유와 직관**으로 풀어 다시 본다. 수학 알레르기를 해체한다.

## 3.1 개요

이 장은 새 개념 없이 2장의 내용을 **다른 각도**에서 다시 본다. 수식이 익숙하지 않다면 이 장을, 수식이 이미 편하면 4장으로 건너뛴다.

## 3.2 비유 1 — Forward = 카프카 큐에 노이즈 메시지 끼워넣기

### 3.2.1 시나리오

당신은 결제 시스템 백엔드 엔지니어다. 카프카에 흐르는 결제 이벤트 데이터에서 의도적으로 **잡음을 끼워넣어** 시스템 견고성을 테스트해야 한다.

```
매 1초마다:
  새 데이터 = (1 - 0.01) * 이전데이터 + 0.01 * 랜덤_노이즈
```

이걸 1000초 (1000 step) 반복하면 데이터는 **거의 원본을 잃어버리고 노이즈가 된다**.

이게 **Forward process**다.

### 3.2.2 수식 매핑

| 수식 | 결제 시스템 비유 |
|---|---|
| $x_0$ | 원본 결제 이벤트 |
| $x_t$ | t초 후의 변형된 이벤트 |
| $\beta_t$ | 매 초 끼워넣는 노이즈 비율 |
| $\epsilon$ | 그날의 랜덤 잡음 |

## 3.3 비유 2 — Reverse = 망가진 데이터에서 원본 복원

### 3.3.1 시나리오

며칠 후, 누군가 당신에게 "**1000초가 지난 망가진 데이터에서 원본을 복원**"해달라고 한다.

너무 어려우니 한 번에 못 복원한다. 대신:
1. 999초 후 상태로 한 단계 거슬러 가기.
2. 998초 후 상태로 또 한 단계 거슬러 가기.
3. ... (1000번 반복)
4. 마침내 0초 (원본) 도달.

매 한 단계는 "**지금 들어있는 노이즈가 얼마나 되는지**"를 추측해서 빼면 된다.

이게 **Reverse process**다.

### 3.3.2 신경망의 역할

복원 매 단계에서 **"지금 들어있는 노이즈"**를 예측하는 함수가 필요하다. 그 함수를 **신경망 $\epsilon_\theta$ 가 학습** 한다.

학습은 어떻게? Forward로 의도적으로 망친 데이터를 만든 다음, "**원래 끼워넣은 노이즈가 얼마였는지**"를 정답으로 주고 신경망에게 맞추게 한다.

이게 DDPM 학습의 본질이다. **"내가 망쳤으니 정답을 알고 있다"** 가 핵심.

## 3.4 비유 3 — Noise Schedule = 운영 정책

### 3.4.1 시나리오

매 step의 노이즈 비율 $\beta_t$ 를 어떻게 정할 것인가?

| 정책 | 의미 |
|---|---|
| Linear schedule | $\beta_t$ 가 시간에 따라 직선으로 증가 |
| Cosine schedule | 매끄러운 곡선 |
| Karras schedule | 고품질 schedule |

이건 **TTL 정책 / 스케줄러 정책 결정**과 같다. 어떤 정책이 더 좋은지는 도메인에 따라 다르고, 실험으로 결정한다.

## 3.5 핵심 원리 — 왜 신경망이 이걸 배울 수 있는가

직관:
- Forward는 **결정론적 + 가우시안**: 수학적으로 완전히 정의됨.
- 그러므로 임의의 $(x_t, t)$ 에 대해 **"무슨 노이즈가 들어있는지"** 가 의미 있게 정의됨.
- 신경망은 충분히 큰 데이터셋으로 이 패턴을 학습 가능.

이게 GAN에는 없는 **수학적 안정성**의 원천이다.

## 3.6 시각자료 — 카프카 비유 그림

```
시간 0초:    [원본 결제 이벤트]
              │
              ▼  + 1% 노이즈
시간 1초:    [거의 원본 + 살짝 노이즈]
              │
              ▼  + 1% 노이즈
시간 2초:    [원본 + 노이즈 더 많음]
              │
              ▼
              ...
              │
              ▼
시간 1000초: [거의 순수 노이즈]


복원할 때 (모델이 학습되어 있다고 가정):

시간 1000초: [거의 순수 노이즈]
              │ ← 모델이 "여기 들어있는 노이즈" 예측해서 빼줌
              ▼
시간 999초:  [노이즈가 살짝 줄어든 데이터]
              │
              ▼
              ...
              │
              ▼
시간 0초:    [거의 원본 복원]
```

## 3.7 코드 — DDPM 학습/추론을 NumPy로

NumPy만으로 1D 데이터에 대해 DDPM을 구현해본다 (60줄).

```python
# practice_3_ddpm_1d.py
import numpy as np

T = 100
betas = np.linspace(0.0001, 0.02, T)
alphas = 1 - betas
alpha_bar = np.cumprod(alphas)

# 1D 가짜 데이터 (sin 곡선)
x_0 = np.sin(np.linspace(0, 2*np.pi, 64))

# Forward
def forward(x_0, t):
    noise = np.random.randn(*x_0.shape)
    x_t = np.sqrt(alpha_bar[t]) * x_0 + np.sqrt(1 - alpha_bar[t]) * noise
    return x_t, noise

# 시각화 (텍스트)
for t in [0, 10, 50, 99]:
    x_t, _ = forward(x_0, t)
    print(f"t={t}: range=[{x_t.min():.2f}, {x_t.max():.2f}]")

# 출력 예:
# t=0: range=[-1.00, 1.00]
# t=10: range=[-1.20, 1.30]
# t=50: range=[-2.50, 2.30]
# t=99: range=[-3.10, 2.80]   ← 거의 순수 가우시안
```

## 3.8 자가 점검

1. Forward process를 카프카 비유로 한 문장으로 설명하라.
2. 신경망이 예측하는 것은 "원본 이미지" 인가, "지금의 노이즈" 인가?
3. DDPM 학습이 GAN보다 안정적인 이유는?
4. Linear schedule과 Cosine schedule은 무엇이 다른가?
5. 위 3.7 코드에서 t를 0~99로 늘리면 데이터 분포가 어떻게 변하는가?

---

# 4장. DDIM — 1000 step을 50 step으로

## 🎯 이 장의 목적

**DDIM (Denoising Diffusion Implicit Models, 2020 말)** 이 어떻게 같은 모델로 **20배 적은 step**에 도달했는지 이해한다.

## 4.1 개요

DDPM은 강력했지만 1000 step이 너무 느렸다. 같은 **모델 학습은 그대로 두고**, **샘플링 알고리즘만 바꿔** 50~100 step으로 줄인 것이 DDIM.

이게 디퓨전을 **실용 가능한 영역**으로 끌어들인 첫 걸음이다.

## 4.2 기초 — DDPM의 sampling이 왜 1000 step이어야 했는가

### 4.2.1 DDPM의 stochastic sampling

DDPM의 reverse는 **확률적**이다.

$$
x_{t-1} = \mu_\theta(x_t, t) + \sigma_t \cdot \epsilon
$$

매 step마다 **랜덤 노이즈를 추가**한다. 이 추가가 있어야 분포가 잘 회복된다.

이 stochasticity 때문에 step을 건너뛰면 **분포가 왜곡**되어 품질 폭락.

### 4.2.2 직관적 비유

매 step에 동전을 던지면서 진행하는 길. 1000 step을 50 step으로 줄이면 **확률적 잡음이 누적**되어 결과가 어긋난다.

## 4.3 핵심 원리 — DDIM의 결정적 단순화

### 4.3.1 핵심 아이디어

> **DDPM의 stochastic sampling을 결정론적(deterministic) sampling으로 바꾼다.**

즉, 매 step에서 **랜덤 노이즈를 더하지 않는다**. 같은 입력에 같은 출력.

### 4.3.2 왜 이게 step을 줄일 수 있게 하는가

결정론적이 되면:
- 매 step의 동작이 **수학적으로 한 ODE 궤적**.
- 이 궤적을 **큰 step으로 점프**해도 어긋남이 작다 (Euler method).
- 따라서 **20~50 step**으로도 같은 품질 도달 가능.

### 4.3.3 같은 모델 재사용

핵심 포인트:
- **모델 재학습 불필요**.
- DDPM 으로 학습한 weight 그대로 사용.
- **샘플링 알고리즘만** 교체.

이 단순함이 DDIM이 빠르게 보급된 이유.

## 4.4 자세히 — DDIM의 수식

### 4.4.1 한 step의 업데이트

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1}} \cdot \epsilon_\theta(x_t, t)
$$

여기서 $\hat{x}_0$ 는 현재 시점에서 모델이 예측하는 "원본 이미지" 추정.

$$
\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}
$$

### 4.4.2 코드로

```python
def ddim_step(x_t, t, t_prev, model, alpha_bar):
    # 1. 모델로 노이즈 예측
    eps = model(x_t, t)
    
    # 2. 원본 추정
    x_0_hat = (x_t - torch.sqrt(1 - alpha_bar[t]) * eps) / torch.sqrt(alpha_bar[t])
    
    # 3. t_prev 시점의 x 직접 계산
    x_prev = (
        torch.sqrt(alpha_bar[t_prev]) * x_0_hat +
        torch.sqrt(1 - alpha_bar[t_prev]) * eps
    )
    return x_prev

# 50 step sampling
def sample_ddim(model, shape, num_steps=50):
    x = torch.randn(shape)
    
    timesteps = torch.linspace(999, 0, num_steps + 1).long()
    for i in range(num_steps):
        t = timesteps[i]
        t_prev = timesteps[i+1]
        x = ddim_step(x, t, t_prev, model, alpha_bar)
    
    return x
```

50번만 model을 호출한다. **20배 빠르다**.

## 4.5 한 걸음 더 — η 파라미터

DDIM은 사실 **stochastic ↔ deterministic 사이의 연속체**를 정의한다.

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \epsilon_\theta + \sigma_t \cdot \epsilon
$$

- $\eta = 0$: 순수 결정론적 (DDIM).
- $\eta = 1$: 완전 stochastic (DDPM과 동등).

이 사이 어디든 선택 가능.

## 4.6 시각자료 — DDPM vs DDIM

```
DDPM (1000 step):
[noise] → step → step → step → ... → step → [image]
                  (1000번 반복, 매번 random noise 추가)

DDIM (50 step):
[noise] → BIG step → BIG step → ... → BIG step → [image]
                  (50번 반복, deterministic)
```

## 4.7 핵심 정리

```
┌────────────────────────────────────────────────────────┐
│  DDIM의 본질:                                           │
│  "stochastic sampling을 deterministic ODE로 바꿔서      │
│   큰 step을 가능하게 만든다."                            │
│                                                          │
│  → 모델 재학습 없음                                      │
│  → 20배 가속                                             │
│  → 품질 거의 동일                                        │
│                                                          │
│  이후 DPM-Solver, UniPC 등 더 영리한 ODE solver들이     │
│  나와서 step을 더 줄임 (10~20 step 영역까지)            │
└────────────────────────────────────────────────────────┘
```

## 4.8 자가 점검

1. DDPM 1000 step을 그냥 50 step으로 건너뛰면 왜 안 되는가?
2. DDIM이 step을 줄일 수 있게 만든 결정적 변경은?
3. DDIM은 DDPM 모델을 다시 학습해야 하는가?
4. DDIM의 한 step 업데이트 식에서 모델은 무엇을 예측하는가?
5. $\eta$ 파라미터의 의미는?

---

# 5장. Latent Diffusion — Stable Diffusion 폭발

## 🎯 이 장의 목적

**Stable Diffusion이 왜 폭발적으로 보급되었는가**의 결정적 이유 — Latent Diffusion 아이디어를 이해한다.

## 5.1 개요

지금까지의 디퓨전은 **픽셀 공간**에서 작동했다. 1024×1024 이미지면 약 300만 차원. 너무 무거움.

**Latent Diffusion (Rombach et al., 2022, "Stable Diffusion")** 은 이 무게를 풀었다:

> **VAE로 이미지를 작은 latent로 압축한 뒤, 그 latent에서 디퓨전한다.**

이 한 발상으로:
- 메모리 12배+ 절약.
- 속도 큰 폭 가속.
- 일반 GPU에서도 학습/추론 가능.

→ Stable Diffusion 출시 후 **수개월 안에 수백만 명**이 디퓨전을 손에 쥐었다.

## 5.2 기초 — 픽셀 디퓨전의 한계

### 5.2.1 메모리

1024×1024×3 = 약 314만 차원의 텐서.

| Step | 메모리 (FP16) |
|---|---|
| 한 이미지 | 6MB |
| Batch 8 | 50MB |
| Attention activation | 추가 수 GB |

→ 한 GPU에 batch 1조차 어려움.

### 5.2.2 속도

각 step마다 백만+ 픽셀에 대해 UNet/DiT 통과.

DALL-E 2, Imagen 같은 픽셀 디퓨전:
- A100에서 한 장 ~30초.
- 학습은 수천 GPU × 수개월.

### 5.2.3 결론

**일반 사용자가 못 쓴다**. 연구실 + 빅테크 전유물.

## 5.3 핵심 원리 — Latent Diffusion의 발상

### 5.3.1 핵심 트릭

```
[1024×1024 이미지] ──VAE encode──→ [128×128×4 latent]   (12배 작음)
                                              │
                                              ▼
                                    [latent 공간에서 디퓨전]
                                              │
                                              ▼
                                    [denoised latent]
                                              │
                                              ▼
                              [VAE decode] ──→ [1024×1024 이미지]
```

### 5.3.2 왜 작동하는가

VAE의 latent space는 **의미 있는 정보를 보존**한다.
- 픽셀의 디테일(노이즈, 정확한 색)은 버린다.
- 의미 있는 구조(형태, 색조, 조명)는 유지.

이 latent에서 디퓨전을 해도 **결과 이미지는 충분히 좋다** (실험으로 검증됨).

### 5.3.3 12배 가속의 산수

1024×1024 → 128×128 (8배 다운샘플).

각 차원이 8배 줄면:
- 픽셀 수: 64배 감소 (8 × 8).
- 채널 수: 4 (대신 3에서 4로 약간 늘었지만 무시).
- **유효 데이터 양: ~16~64배 감소**.

UNet/DiT의 비용은 데이터 크기에 거의 비례하므로 **속도 5~20배** 가속.

## 5.4 자세히 — 학습과 추론

### 5.4.1 학습 단계

**Stage 1**: VAE 학습 (한 번만, 모든 이미지 도메인에 재사용).
```
이미지 ──Encoder──→ latent ──Decoder──→ 복원 이미지
                       │
                       └─── 복원 손실 + KL loss
```

**Stage 2**: latent 위에서 디퓨전 학습.
- VAE는 freeze.
- latent에 노이즈 더하고 → UNet/DiT가 노이즈 예측.

### 5.4.2 추론 단계

```python
# Latent diffusion 추론 (가짜 코드)
def latent_diffusion_inference(prompt, vae, denoiser, scheduler):
    text_emb = text_encoder(prompt)
    
    # 1. 노이즈 latent 초기화
    latent = torch.randn(1, 4, 128, 128)
    
    # 2. denoise (latent 공간에서)
    for t in scheduler.timesteps:
        noise_pred = denoiser(latent, t, text_emb)
        latent = scheduler.step(noise_pred, t, latent).prev_sample
    
    # 3. latent → 이미지
    image = vae.decode(latent)
    return image
```

이 흐름이 Stable Diffusion 1.x, 2.x, SDXL, FLUX, **이 챌린지의 FLUX.2 Klein** 까지 모두 동일하다.

## 5.5 한 걸음 더 — VAE의 trade-off

### 5.5.1 어떤 디테일은 잃는다

VAE는 **압축이 필연적으로 정보 손실**.
- 작은 텍스트, 가는 선이 약간 흐려짐.
- 사람 얼굴의 미세한 디테일이 미묘하게 변형됨.

### 5.5.2 왜 그래도 받아들이는가

**12배 가속**이 너무 크다. 약간의 디테일 손실은 받아들일 만하다.
또, VAE를 더 좋게 만들면(z_channels 늘리기, 더 깊은 모델) 손실이 줄어든다.

### 5.5.3 FLUX의 VAE 개선

FLUX 시리즈는 SD보다 더 강한 VAE 사용:
- **z_channels = 16 (SD는 4)**.
- 더 많은 정보 보존.
- 그래서 디테일이 더 살아남.

이 챌린지의 FLUX.2 Klein은 **z_channels = 32**. 한 단계 더 강함.

## 5.6 시각자료 — 픽셀 vs Latent

```
픽셀 디퓨전 (옛날):
[1024 × 1024 × 3 = 3.1M dims] ─ denoise ×1000 ─ [1024 × 1024 × 3]
                          ▲
                          무거움, 느림


Latent 디퓨전 (Stable Diffusion+):
[1024 × 1024 × 3]
        │
        ▼ VAE encode
[128 × 128 × 16 ~ 32 = ~260K~520K dims]
        │
        ▼ denoise ×4~50  (12배 빠름)
[128 × 128 × 16~32]
        │
        ▼ VAE decode
[1024 × 1024 × 3]
```

## 5.7 핵심 정리

```
┌──────────────────────────────────────────────────────┐
│ Latent Diffusion = "디퓨전을 압축된 공간에서 한다"   │
│                                                       │
│ 효과: 메모리 12배+ 절약, 속도 5~20배 가속            │
│ 비용: 약간의 디테일 손실 (받아들일 만함)              │
│                                                       │
│ → Stable Diffusion (SD), SDXL, SD3, FLUX 시리즈,     │
│   이 챌린지의 FLUX.2 Klein 모두 latent diffusion     │
└──────────────────────────────────────────────────────┘
```

## 5.8 자가 점검

1. 픽셀 디퓨전의 메모리·속도 한계는?
2. Latent diffusion의 핵심 트릭을 한 문장으로.
3. VAE 학습은 디퓨전 학습과 동시에 하는가, 별도인가?
4. Latent diffusion이 잃는 것은 무엇인가?
5. FLUX.2 Klein의 z_channels는 SD 대비 얼마나 더 많은가?

---

# 6장. Rectified Flow — 직선 경로

## 🎯 이 장의 목적

**Rectified Flow** 가 어떻게 디퓨전의 곡선 경로를 직선으로 펴서 **4 step**까지 줄였는지 이해한다. 이 챌린지의 FLUX.2 Klein이 정확히 이 방식이다.

## 6.1 개요

DDPM (1000 step) → DDIM (50 step) 까지는 **샘플링만 영리하게** 한 가속이었다.

Rectified Flow는 **모델 자체를 다시 학습**해서 더 큰 가속을 가능하게 했다.

> **곡선 경로(curved trajectory)를 직선으로 펴면 더 큰 보폭(step)으로 갈 수 있다.**

## 6.2 기초 — 디퓨전의 경로가 왜 곡선인가

### 6.2.1 시각화

DDPM의 reverse process를 1차원으로 시각화하면:

```
시작점 (노이즈)
   │
   ╲
    ╲
     ╲╲                    ← 곡선 경로
       ╲╲
         ╲╲
           ╲___
               ╲___
                   ╲___ → 도착점 (이미지)
```

이 길이 휘어 있어서 **큰 보폭으로 가면 길에서 벗어남**.

### 6.2.2 ODE의 관점

Reverse process는 본질적으로 ODE (Ordinary Differential Equation):

$$
\frac{dx}{dt} = v_\theta(x, t)
$$

여기서 $v_\theta$ 는 "이 점에서 어디로 가야 하는지" 의 방향.

만약 $v_\theta$ 가 **위치마다 크게 변하면** (curved field), 큰 step은 부정확.
만약 $v_\theta$ 가 **거의 일정하면** (straight field), 큰 step도 OK.

## 6.3 핵심 원리 — Rectified Flow의 발상

### 6.3.1 핵심 아이디어

> **모델을 처음부터 "직선 경로의 velocity" 를 학습하도록 가르친다.**

학습 시:
- 샘플 $x_0$ (이미지) 와 $x_1$ (노이즈) 를 잇는 **직선** 을 정의.
- 그 직선 위 임의의 점 $x_t = (1-t) x_0 + t x_1$.
- 그 점에서의 velocity = $x_1 - x_0$ (직선이라 일정!).
- 모델이 이 velocity 를 예측.

학습 손실:
$$
\mathcal{L} = \mathbb{E}_{t, x_0, x_1} \left[ \| (x_1 - x_0) - v_\theta(x_t, t) \|^2 \right]
$$

### 6.3.2 왜 이게 직선 경로를 만드는가

학습된 $v_\theta$ 는 **모든 곳에서 같은 방향(노이즈→데이터)을 가리키도록 강요받는다**.
이게 **직선 경로**를 만든다.

### 6.3.3 추론 (sampling)

직선이라 큰 step도 OK:

```python
def sample_rectified_flow(model, shape, num_steps=4):
    x = torch.randn(shape)  # 노이즈에서 시작
    
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = 1.0 - i * dt
        v = model(x, t)
        x = x - v * dt   # Euler step (큰 step OK)
    
    return x  # 이미지
```

**4 step만으로 이미지 완성**.

## 6.4 자세히 — Reflow

### 6.4.1 한 번 더 직선화

Rectified Flow를 한 번 학습한 모델은 이미 직선에 가깝지만, **완벽하지 않다**.

**Reflow**: 학습된 모델로 만든 (노이즈, 이미지) 쌍을 가지고 **다시 학습**.
- 첫 번째 학습: 무작위 (노이즈, 이미지) 매칭에서 직선 학습.
- 두 번째 학습 (Reflow): 모델이 만든 매칭 쌍에서 더 정밀한 직선 학습.

이렇게 하면 **점점 더 직선화**.

### 6.4.2 1-step까지 가능

극단적으로 reflow + distillation을 결합하면 **1 step**도 가능 (SD3 Turbo, FLUX.1 schnell 등).

## 6.5 한 걸음 더 — Flow Matching 일반론

Rectified Flow는 더 큰 패러다임 **Flow Matching (Lipman et al., 2022)** 의 특수 케이스.

| 방식 | 학습 대상 | 경로 |
|---|---|---|
| DDPM | $\epsilon_\theta$ (노이즈) | 곡선 |
| Score matching | $\nabla \log p$ (점수) | 곡선 |
| Rectified Flow | $v_\theta$ (velocity) | 직선 |
| Flow Matching | $v_\theta$ | 임의 (선택 가능) |

핵심은 **"각 점에서 어디로 가는지"의 vector field를 학습** 한다는 것. 어떤 vector field를 선택하느냐가 차이.

## 6.6 시각자료 — 곡선 vs 직선

```
DDPM (curved):                    Rectified Flow (straight):

[noise]                           [noise]
  ╲                                  │
   ╲                                  │
    ╲                                  │ (직선)
     ╲                                  │
      ╲                                  │
       ╲___                              │
           ╲___                          │
               ╲___ [image]              ▼
              (50~1000 step 필요)     [image]
                                     (4 step OK)
```

## 6.7 코드 — 1D 토이 예제

```python
# practice_6_rectified_flow_1d.py
import torch
import torch.nn as nn

class TinyVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x, t):
        # t를 input에 concat
        if t.dim() == 0: t = t.expand(x.shape[0])
        inp = torch.cat([x, t.unsqueeze(-1)], dim=-1)
        return self.net(inp)

# 가짜 데이터: 1차원 가우시안 → 1차원 가우시안 (다른 평균)
def get_data(batch_size):
    x_0 = torch.randn(batch_size, 1) + 5.0  # 데이터: 평균 5
    x_1 = torch.randn(batch_size, 1)          # 노이즈: 평균 0
    return x_0, x_1

model = TinyVelocity()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# 학습
for step in range(2000):
    x_0, x_1 = get_data(256)
    t = torch.rand(256)
    x_t = (1 - t.unsqueeze(-1)) * x_0 + t.unsqueeze(-1) * x_1
    target_v = x_1 - x_0
    pred_v = model(x_t, t)
    loss = ((pred_v - target_v) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0:
        print(f"step {step}: loss {loss.item():.4f}")

# 추론 (4 step)
model.eval()
with torch.no_grad():
    x = torch.randn(1, 1)  # 노이즈에서 시작
    for i in range(4):
        t = torch.tensor(1.0 - i * 0.25)
        v = model(x, t)
        x = x - v * 0.25
    print(f"4-step result: {x.item():.4f}  (target ≈ 5)")
```

이 60줄 코드에 **Rectified Flow의 본질**이 모두 들어있다.

## 6.8 자가 점검

1. 디퓨전의 reverse 경로가 곡선인 이유는?
2. Rectified Flow가 학습하는 양은 무엇인가?
3. Velocity가 일정하면 큰 step이 가능한 이유는?
4. Reflow는 무엇이며 왜 하는가?
5. 위 6.7 토이 코드의 step을 1로 줄여보면 결과가 얼마나 달라지는가?

---

# 7장. Distillation — Teacher의 50 step을 Student의 4 step으로

## 🎯 이 장의 목적

**지식 증류(distillation)** 가 어떻게 큰/느린 모델의 능력을 작은/빠른 모델로 옮기는지 이해한다. FLUX.2 Klein이 4 step인 이유가 distillation 임을 안다.

## 7.1 개요

Rectified Flow도 강력하지만, **추가 distillation** 을 결합하면 더 줄일 수 있다.

> **Teacher (예: 50 step Rectified Flow) 가 만든 결과를 Student (4 step) 가 흉내내도록 학습.**

## 7.2 기초 — Knowledge Distillation 일반론

### 7.2.1 기본 개념 (Hinton et al., 2015)

큰 모델 (Teacher) 의 출력을 작은 모델 (Student) 의 학습 정답으로 사용.

```
입력 X
   │
   ├──→ [Teacher] → soft label (예: 0.7, 0.2, 0.1)
   │
   └──→ [Student] → 예측 (0.6, 0.3, 0.1)
                          │
                          ▼
                    Teacher의 soft label과 일치하도록 학습
```

### 7.2.2 왜 작동하는가

Hard label ("이 사진은 강아지") 보다 soft label ("70% 강아지, 20% 고양이, 10% 늑대") 이 **더 풍부한 정보**.

Student가 이 풍부한 정보로 학습하면 **Teacher의 미묘한 판단까지** 흡수.

## 7.3 핵심 원리 — Diffusion Distillation

### 7.3.1 디퓨전에 적용

Teacher: 50 step DDIM 또는 Rectified Flow.
Student: 4 step (또는 1 step) 디퓨전.

학습:
- 같은 입력 노이즈 $x_T$ 에서 시작.
- Teacher가 50 step 거쳐 $x_0^{teacher}$ 생성.
- Student가 4 step 거쳐 $x_0^{student}$ 생성.
- 두 결과가 일치하도록 student 학습.

### 7.3.2 점진적 distillation (Progressive Distillation)

한 번에 50 → 4 가는 게 아니라:

```
50 step → 25 step → 12 step → 6 step → 4 step
```

각 단계에서 절반씩 줄이는 distillation을 반복.

이게 **Salimans & Ho, 2022 ("Progressive Distillation")** 의 발명.

### 7.3.3 다른 distillation 패밀리

| 방법 | 핵심 |
|---|---|
| Progressive Distillation | 절반씩 단계적 |
| Consistency Models (Song, 2023) | 어디서 출발하든 같은 결과로 수렴 |
| LCM (Latent Consistency Model) | SD에 Consistency 적용 |
| SDXL Turbo | Adversarial distillation (GAN loss) |
| ADD (Adversarial Diffusion Distillation) | 1 step 가능 |
| FLUX.1 schnell | 1~4 step distillation |
| **FLUX.2 Klein [4 step]** | **이 챌린지의 모델** |

## 7.4 자세히 — Distillation의 trade-off

### 7.4.1 가속 폭

- 50 → 4 step = **12.5배** 가속.
- 1 step 모델 = **추가 4배** = 50배.

### 7.4.2 품질 손실

Distillation 모델은:
- **다양성 약간 감소** (teacher의 평균을 따라가는 경향).
- **세밀한 디테일 약간 손실**.

서비스 관점에서 이 trade-off는 받아들일 만하다.

### 7.4.3 학습 비용

Distillation 학습은 **Teacher 추론 + Student 학습** 이라 시간이 든다.

이미 학습된 distilled 모델을 받아 쓰는 게 보통.

## 7.5 시각자료 — Distillation 흐름

```
[학습 단계]

x_T (노이즈)
   │
   ├──→ [Teacher (50 step)] ─→ x_0_teacher
   │
   └──→ [Student (4 step)]  ─→ x_0_student
                                       │
                                       │
   x_0_teacher와 x_0_student의 차이를  │
   loss로 사용해 student 학습          │
                                       ▼
                                    [Loss]


[Progressive Distillation]

[1024 step] → distill → [512 step] → distill → [256 step] → ...
                                                              ↓
                                                          [4 step]
                                                              ↓
                                                          [1 step]
```

## 7.6 핵심 정리

```
┌──────────────────────────────────────────────────────┐
│ Distillation = "큰/느린 모델 → 작은/빠른 모델로 압축" │
│                                                        │
│ → FLUX.2 Klein의 4 step은 distillation의 결과물       │
│ → 이미 가속의 큰 부분이 모델 자체에 흡수됨             │
│ → 이 챌린지가 추가 가속을 요구하는 이유:               │
│   "쉬운 가속은 이미 끝났으니 더 깊은 곳에서 찾아라"   │
└──────────────────────────────────────────────────────┘
```

## 7.7 자가 점검

1. Knowledge distillation의 한 줄 정의는?
2. Soft label이 hard label보다 풍부한 이유는?
3. Progressive Distillation의 핵심 발상은?
4. Distillation의 두 가지 품질 trade-off는?
5. FLUX.2 Klein의 4 step은 distillation의 어떤 결과인가?

---

# 8장. Classifier-Free Guidance (CFG)

## 🎯 이 장의 목적

**CFG가 무엇이며 왜 매 step 두 번 도는지**, 그리고 **FLUX.2 Klein이 CFG=1.0 이라는 의미**를 정확히 안다.

## 8.1 개요

순수 디퓨전은 텍스트 프롬프트를 약하게 따른다. **CFG는 프롬프트 따르기 강도를 조절**하는 추론 시점 트릭이다.

```
강아지 프롬프트
   │
   ▼
CFG 적용 X: 강아지 비슷한 동물이 나옴
CFG 적용 O: 정확히 강아지가 나옴 (강도 조절 가능)
```

## 8.2 기초 — Classifier Guidance (CFG의 전신)

### 8.2.1 원래 방법 (2021)

별도의 분류기를 학습해서, 그 분류기의 그래디언트를 디퓨전 reverse에 추가:

```
ε_guided = ε_model + scale × ∇log p_classifier(class | x_t)
```

### 8.2.2 한계

- **분류기를 따로 학습해야 함**.
- **OOD(분포 밖) 데이터에 약함**.

## 8.3 핵심 원리 — CFG (Ho & Salimans, 2021)

### 8.3.1 핵심 발상

> **분류기 없이, 모델을 두 번 호출해서 차이를 이용한다.**

학습 시:
- 일정 확률로 **텍스트 임베딩을 비움** (unconditional 학습).
- 같은 모델이 **conditional + unconditional** 둘 다 학습됨.

추론 시:
```python
eps_uncond = model(x_t, t, text_emb=null)
eps_cond   = model(x_t, t, text_emb=prompt)

eps_guided = eps_uncond + scale * (eps_cond - eps_uncond)
```

이 `scale` 이 CFG scale (보통 7.0).

### 8.3.2 직관

`(eps_cond - eps_uncond)` = "프롬프트가 만드는 차이의 방향".
이 방향으로 더 강하게 밀면 → 프롬프트가 더 강조됨.

### 8.3.3 비용 — 매 step 모델 두 번 호출

```
일반 디퓨전 1 step = 1 forward
CFG 디퓨전 1 step = 2 forward (cond + uncond)

→ 추론이 2배 느려짐
```

**Batch 트릭**: cond와 uncond를 batch=2로 묶어 한 번에 호출하면 2배 효율.

## 8.4 자세히 — FLUX.2 Klein의 CFG=1.0

### 8.4.1 수식의 의미

```
eps_guided = eps_uncond + 1.0 * (eps_cond - eps_uncond)
           = eps_cond
```

CFG scale = 1.0 이면 **uncond는 사용되지 않음** = 사실상 CFG 없음.

### 8.4.2 왜 이렇게 됐는가

FLUX.2 Klein은 **distillation 과정에서 CFG 효과를 모델 안에 흡수**:
- Teacher: 일반 모델 + CFG (scale=7.0).
- Student: CFG 없이 한 번만 호출해도 같은 결과.

이걸 **CFG distillation** 이라 부른다.

### 8.4.3 추론 함의

- **한 step = 한 forward** (CFG가 있는 일반 디퓨전은 cond + uncond 합쳐 forward 두 번 = batch 2배).
- **이미 batch=1 추론** → 단순 batch 합치기 같은 트릭은 못 씀.
- **모델 자체의 효율성**, **블록·토큰 수준의 캐싱** 같은 더 깊은 최적화로 가야 함.

이게 챌린지가 "쉬운 답을 거부한다" 고 하는 이유.

## 8.5 한 걸음 더 — CFG의 부작용

### 8.5.1 너무 큰 scale의 문제

scale = 15+ 같은 값은:
- 색이 과포화됨.
- 디테일이 부서짐.
- "Burned" 느낌.

### 8.5.2 동적 CFG (Dynamic CFG)

step마다 scale을 다르게:
- 초반 step: 큰 scale (구도 잡기).
- 후반 step: 작은 scale (디테일 살리기).

이 챌린지에는 직접 관련 적음.

## 8.6 시각자료 — CFG 효과

```
[noise] ──┐
          ▼
       [model] → ε_uncond  (조건 없음)
          
[noise] ──┐
          ▼
[prompt]──┴→[model] → ε_cond  (조건 있음)


eps_guided = ε_uncond + scale × (ε_cond - ε_uncond)
                              ↑
                     이 차이 벡터가 "프롬프트 효과"
                     scale이 클수록 강조됨


시각:
  ε_uncond   ●
              │
              │ × scale
              ▼
  ε_cond ─────●  ← 이 방향으로 더 가기
              ↓
        ε_guided ●
```

## 8.7 핵심 정리

```
┌────────────────────────────────────────────────────────┐
│ CFG = "분류기 없이 모델을 두 번 호출해서 프롬프트       │
│        따르기 강도를 조절하는 추론 트릭"                 │
│                                                          │
│ 비용: 추론 2배 느려짐 (cond + uncond)                    │
│                                                          │
│ FLUX.2 Klein: CFG distillation으로 이미 흡수            │
│   → CFG scale = 1.0 = CFG 없음                           │
│   → 한 step = 한 forward (배치=1 추론)                   │
│                                                          │
│ 함의: 단순 가속 카드는 이미 사용됨                       │
│       → 더 깊은 최적화 (캐싱, 양자화) 필요               │
└────────────────────────────────────────────────────────┘
```

## 8.8 자가 점검

1. Classifier Guidance와 Classifier-Free Guidance의 차이는?
2. CFG가 매 step 모델을 두 번 호출하는 이유는?
3. CFG scale = 1.0의 수식적 의미는?
4. FLUX.2 Klein이 CFG=1.0인 이유와 그 함의는?
5. CFG scale을 너무 크게 하면 무엇이 문제인가?

---

# 9장. 디퓨전 추론 한 cycle 코드 분해

## 🎯 이 장의 목적

지금까지 배운 모든 조각을 **한 코드 흐름**으로 결합한다. SD/FLUX 추론의 정확한 9단계를 다시 본다.

## 9.1 개요

디퓨전 추론 1 cycle은 다음 9 단계.

```
[1] 입력: 텍스트 + (옵션) 레퍼런스 이미지
[2] 텍스트 인코딩 (한 번만)
[3] (옵션) 레퍼런스 이미지 → VAE encode → reference latent
[4] 노이즈 latent 초기화
[5] Denoising loop (4~50 step 반복)
    [5a] 모델 forward (text emb + ref latent + noise latent + timestep)
    [5b] 노이즈 또는 velocity 예측
    [5c] Scheduler로 한 step 업데이트
[6] 최종 latent → VAE decode
[7] 후처리 (denormalize, clip, uint8 변환)
[8] 결과 이미지 반환
```

## 9.2 기초 — 코드로 분해

### 9.2.1 Stable Diffusion 1.5 — 전체 추론 코드

```python
# practice_9_sd_full.py
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

# 자동: pipe(prompt) 한 줄
image = pipe("a photograph of an astronaut on a horse", num_inference_steps=20).images[0]
image.save("auto.png")

# 수동 분해 ↓
prompt = "a photograph of an astronaut on a horse"
device = "cuda"

# [2] 텍스트 인코딩
text_input = pipe.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
text_emb = pipe.text_encoder(text_input)[0]
uncond_input = pipe.tokenizer("", return_tensors="pt").input_ids.to(device)
uncond_emb = pipe.text_encoder(uncond_input)[0]
text_emb_combined = torch.cat([uncond_emb, text_emb])  # CFG용

# [4] 노이즈 latent
generator = torch.manual_seed(0)
latents = torch.randn(
    (1, 4, 64, 64), generator=generator, device=device, dtype=torch.float16
)

# [5] Denoising
pipe.scheduler.set_timesteps(20)
guidance_scale = 7.5

for t in pipe.scheduler.timesteps:
    # CFG: latent를 batch=2로 복제
    latent_input = torch.cat([latents] * 2)
    
    with torch.no_grad():
        # [5a] 모델 forward
        noise_pred = pipe.unet(
            latent_input, t, encoder_hidden_states=text_emb_combined
        ).sample
    
    # [5b] CFG 적용
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    
    # [5c] Scheduler step
    latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

# [6] VAE decode
latents = latents / pipe.vae.config.scaling_factor
with torch.no_grad():
    image = pipe.vae.decode(latents).sample

# [7] 후처리
image = (image / 2 + 0.5).clamp(0, 1)
image = (image.cpu().permute(0, 2, 3, 1).numpy() * 255).round().astype("uint8")[0]

# [8] 저장
from PIL import Image
Image.fromarray(image).save("manual.png")
```

이 한 파일에 **9 단계가 모두 들어있다**.

## 9.3 핵심 원리 — 어디서 시간이 가는가

### 9.3.1 시간 분포 (SD 1.5, A100, 20 step)

```
[2] Text encoding:        ~30ms      (1회만)
[4] Noise init:           <1ms
[5] Denoising loop:       ~700ms     (20 × 35ms ≈ 80%) ← 핵심
[6] VAE decode:           ~150ms     (~17%)
[7] 후처리:               ~10ms

총: ~900ms
```

**80% 가 denoising loop**. 그래서 최적화의 중심이 여기.

### 9.3.2 FLUX.2 Klein은?

비슷한 분포지만 step이 **20 → 4** 라 denoising 비중이 줄고 **VAE decode가 상대적으로 큼**.

```
Text encoding:    ~5%
Denoising x4:     ~75%  ← 여전히 가장 큼
VAE decode:       ~15%
```

이 챌린지의 최적화 우선순위:
1. Denoising 가속 (FlashAttention, FP8, caching).
2. VAE 가속 (압축, FP16).
3. Text encoding은 1회라 중요도 낮음.

## 9.4 자세히 — FLUX 코드와 SD 코드의 차이

### 9.4.1 큰 그림은 같다

9 단계 워크플로우는 동일.

### 9.4.2 차이점

| 항목 | SD 1.5 | FLUX.2 Klein |
|---|---|---|
| Backbone | UNet | DiT (Transformer) |
| Text encoder | CLIP | T5 + CLIP |
| Step 수 | 20~50 | 4 (distilled) |
| Sampler | DDIM/DPM-Solver | Rectified Flow ODE |
| CFG | 사용 | 1.0 (없음) |
| Latent z_channels | 4 | 32 |
| Reference image | (없음) | 가능 |

이 차이들이 챌린지의 출제 배경이다.

## 9.5 시각자료 — 추론 흐름과 시간 분포

```
[Prompt + Reference image]
         │
         ▼
   [Text encoder]   <30ms (1회)
         │
         ▼
   [(옵션) Reference VAE encode]   ~30ms
         │
         ▼
   [Noise latent 초기화]
         │
         ▼
┌──────────────────────────────┐
│   Denoising loop (4~50 step) │  ← ~75% of total time
│   ┌─────────────┐             │
│   │ Model forward│             │
│   │ Noise predict│             │
│   │ Scheduler step              │
│   └─────────────┘             │
└──────────────────────────────┘
         │
         ▼
   [VAE decode]   ~15%
         │
         ▼
   [후처리]   <1%
         │
         ▼
   [최종 이미지]
```

## 9.6 자가 점검

1. 디퓨전 추론 1 cycle의 9단계를 입으로 답하라.
2. SD 추론에서 시간의 80%가 어디서 가는가?
3. FLUX.2 Klein은 SD 1.5와 어떤 점이 가장 다른가?
4. 위 9.2.1 코드에서 CFG를 끄려면 어떤 줄을 수정하는가?
5. VAE decode를 캐싱할 수 있는가? 왜?

---

# 10장. 디퓨전 vs LLM — KV Cache가 안 통하는 이유

## 🎯 이 장의 목적

LLM에서는 KV cache가 표준 가속이다. **왜 디퓨전에는 그대로 적용되지 않는가**, 그리고 **이 챌린지의 핵심 통찰**을 이끌어낸다.

## 10.1 개요

LLM (GPT, Claude 등) 의 추론은 **autoregressive** — 한 번에 한 토큰씩 생성. KV cache가 자연스럽게 맞아떨어진다.

디퓨전은 다르다. **매 step에서 모든 토큰의 입력이 바뀐다.**
→ 표준 KV cache는 안 통한다.

그러나 **부분적인 캐싱은 가능하다.** 그게 이 챌린지의 핵심.

## 10.2 기초 — LLM의 KV Cache

### 10.2.1 LLM 추론의 본질

```
"안녕" → 다음 토큰 예측: "하"
"안녕하" → 다음 토큰 예측: "세"
"안녕하세" → 다음 토큰 예측: "요"
"안녕하세요" → 다음 토큰 예측: "."
```

매 step마다 **이전 모든 토큰 + 새 토큰** 을 모델에 입력.

### 10.2.2 순진한 구현의 비효율

```python
# 매번 전체 시퀀스를 처리
for step in range(max_length):
    output = model(tokens[:step+1])
    next_token = output[-1].argmax()
    tokens.append(next_token)
```

100 토큰 생성 시 모델 호출 100번. 매번 이전 토큰들도 다시 attention 처리. **O(n²) 비용 누적**.

### 10.2.3 KV Cache의 해결

**관찰**: 이전 토큰들의 K, V는 **다음 step에서도 변하지 않는다**.

→ 이전 K, V를 **메모리에 저장(cache)** 하고, 새 토큰만 계산.

```python
# KV cache 사용
kv_cache = None
for step in range(max_length):
    output, kv_cache = model.forward_with_cache(new_token, kv_cache)
    next_token = output.argmax()
```

100 토큰 생성 시 모델은 **각 step에서 1 토큰만 처리**. **O(n) 비용**.

이게 vLLM, TGI, TensorRT-LLM 같은 LLM 추론 시스템의 핵심.

## 10.3 핵심 원리 — 디퓨전의 차이

### 10.3.1 매 step 모든 토큰이 바뀐다

디퓨전 reverse:

```
step T:      x_T (전체 노이즈 latent)
step T-1:    x_{T-1} (전체가 변형됨)
...
step 0:      x_0 (이미지)
```

**전체 latent의 모든 픽셀(토큰)이 매 step 변한다.**

→ 토큰의 K, V도 매 step 변한다.
→ 표준 KV cache 불가.

### 10.3.2 직관적 비유

LLM:
- 음식점에서 손님이 한 명씩 추가됨.
- 이미 앉은 손님의 정보(K, V)는 그대로.

디퓨전:
- 매 step마다 모든 손님이 의자를 바꿈.
- 이전 정보가 더 이상 유효하지 않음.

### 10.3.3 그래서 디퓨전은 정말 캐싱 불가능한가

**핵심 질문**: "전체가 바뀌는가?" — 아니다. **부분은 안 바뀐다**.

- **텍스트 토큰의 K, V**: 매 step 동일. → 캐싱 가능.
- **레퍼런스 이미지 토큰의 K, V**: 같은 reference면 동일. → 캐싱 가능 ← 이 챌린지의 핵심!
- **노이즈 latent 토큰의 K, V**: 매 step 변함. → 캐싱 불가.

## 10.4 자세히 — 캐싱 가능 영역의 매핑

### 10.4.1 표 — 디퓨전 토큰별 캐싱 여부

| 토큰 종류 | 변화 여부 | 캐싱 가능 |
|---|---|---|
| 텍스트 토큰 (T5/CLIP 출력) | 같은 prompt면 불변 | ✅ |
| 레퍼런스 이미지 토큰 (VAE encoded) | 같은 ref면 불변 | ✅ ← **이 챌린지의 노다지** |
| 노이즈 latent 토큰 | 매 step 변함 | ❌ |
| Timestep embedding | t에 따라 변함 | step마다 다름 |

### 10.4.2 이 챌린지의 핵심 통찰

> 사용자는 한 세션에서:
> - **아바타는 고정** (한 번 업로드).
> - **옷만 바꿔서 30번 클릭**.
>
> 매 클릭마다:
> - 텍스트는 거의 같다 → 캐싱 가능.
> - **아바타 token의 K, V도 같다 → 캐싱 가능.**
> - 옷 token의 K, V는 새로 계산.
> - 노이즈 latent의 K, V는 매 step 새로 계산.

이걸 **"Reference K/V Caching"** 이라고 부르자. 이 챌린지의 가장 큰 단일 가속 카드.

### 10.4.3 추가 — Step-wise Caching

또 한 종류의 캐싱:
- **인접한 step끼리 모델 출력이 거의 비슷**.
- 일부 layer 출력을 캐시해서 다음 step에서 재사용.

이게 DeepCache, **Learning-to-Cache** (이 챌린지가 직접 추천한 논문).

자세한 내용은 7권에서.

## 10.5 시각자료 — LLM vs 디퓨전 캐싱

```
LLM (autoregressive):
                     ┌────────────────┐
                     │ KV cache       │ ← 이전 토큰들의 K, V
                     │ (매 step 누적)  │
                     └────────────────┘
                              │
                              ▼ ✅ 그대로 재사용
[step N]: 새 토큰 1개만 계산
                              │
                              ▼
                     [다음 토큰 예측]


Diffusion (parallel):
                     ┌────────────────┐
                     │ 표준 KV cache  │ ← 매 step 모두 바뀜
                     │  ❌ 못 씀       │
                     └────────────────┘

                     ┌──────────────────────┐
                     │ Reference token K/V  │ ← 변하지 않는 부분
                     │   ✅ 캐시 가능        │
                     └──────────────────────┘
                     ┌──────────────────────┐
                     │ Noise token K/V      │
                     │   ❌ 매 step 새로     │
                     └──────────────────────┘
```

## 10.6 핵심 정리

```
┌──────────────────────────────────────────────────────────┐
│ 디퓨전이 LLM처럼 표준 KV cache를 못 쓰는 이유:           │
│ "매 step 전체 latent가 바뀌기 때문"                       │
│                                                            │
│ 그러나 "변하지 않는 토큰" (text, reference) 의 K/V는      │
│ 캐싱 가능. 이게 챌린지의 핵심 통찰의 한 축.                │
│                                                            │
│ 추가로 인접 step의 모델 중간 출력도 거의 같으므로          │
│ step-wise feature caching이 또 다른 가속 카드.            │
│                                                            │
│ → 이 두 캐싱 기법이 "transformer-side reuse"의 본체       │
└──────────────────────────────────────────────────────────┘
```

## 10.7 자가 점검

1. LLM의 KV cache가 작동하는 이유는?
2. 디퓨전이 표준 KV cache를 못 쓰는 이유는?
3. 그럼에도 캐싱 가능한 토큰 종류 두 가지를 들어라.
4. 이 챌린지의 시나리오에서 가장 큰 캐싱 기회는 무엇인가?
5. Step-wise feature caching이 어떻게 추가 가속 카드가 되는가?

---

# 11장. 실습 — Stable Diffusion 1.5 로컬 추론 + 시간 측정

## 🎯 이 장의 목적

이 권의 모든 개념을 **실제 GPU에서 측정** 한다. 백엔드 직관 (시간 분포, 병목 식별) 을 AI 추론에 적용하는 첫 경험.

## 11.1 환경 준비

```bash
pip install torch diffusers transformers accelerate
# GPU 있다면 CUDA 버전에 맞는 PyTorch 설치
```

GPU 확인:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

GPU 없으면 Google Colab 무료 T4 사용 권장.

## 11.2 Step 1 — 기본 추론

```python
# step1_basic.py
import torch
import time
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

prompt = "a photograph of an astronaut riding a horse"

# Warmup (첫 호출은 컴파일/캐시 때문에 느림)
_ = pipe(prompt, num_inference_steps=20).images[0]

# 측정
torch.cuda.synchronize()
start = time.time()

image = pipe(prompt, num_inference_steps=20).images[0]

torch.cuda.synchronize()
elapsed = time.time() - start

print(f"Total time: {elapsed*1000:.1f} ms")
image.save("step1.png")
```

**예상 출력 (RTX 4090 기준)**:
```
Total time: 850.2 ms
```

## 11.3 Step 2 — 단계별 시간 분포 측정

```python
# step2_breakdown.py
import torch
import time
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

prompt = "a photograph of an astronaut riding a horse"
device = "cuda"

# Warmup
_ = pipe(prompt, num_inference_steps=5).images[0]

# 1. Text encoding
torch.cuda.synchronize(); t0 = time.time()
text_input = pipe.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
text_emb = pipe.text_encoder(text_input)[0]
uncond_input = pipe.tokenizer("", return_tensors="pt").input_ids.to(device)
uncond_emb = pipe.text_encoder(uncond_input)[0]
text_emb_combined = torch.cat([uncond_emb, text_emb])
torch.cuda.synchronize(); t1 = time.time()
print(f"Text encoding:    {(t1-t0)*1000:.1f} ms")

# 2. Noise init
torch.cuda.synchronize(); t0 = time.time()
latents = torch.randn(
    (1, 4, 64, 64), device=device, dtype=torch.float16
)
torch.cuda.synchronize(); t1 = time.time()
print(f"Noise init:       {(t1-t0)*1000:.1f} ms")

# 3. Denoising loop
pipe.scheduler.set_timesteps(20)
guidance_scale = 7.5

torch.cuda.synchronize(); t0 = time.time()
for t in pipe.scheduler.timesteps:
    latent_input = torch.cat([latents] * 2)
    with torch.no_grad():
        noise_pred = pipe.unet(
            latent_input, t, encoder_hidden_states=text_emb_combined
        ).sample
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
torch.cuda.synchronize(); t1 = time.time()
print(f"Denoising (20):   {(t1-t0)*1000:.1f} ms  ({(t1-t0)*1000/20:.1f} ms/step)")

# 4. VAE decode
torch.cuda.synchronize(); t0 = time.time()
latents = latents / pipe.vae.config.scaling_factor
with torch.no_grad():
    image = pipe.vae.decode(latents).sample
torch.cuda.synchronize(); t1 = time.time()
print(f"VAE decode:       {(t1-t0)*1000:.1f} ms")
```

**예상 출력 (RTX 4090, FP16)**:
```
Text encoding:    25.3 ms
Noise init:       0.4 ms
Denoising (20):   620.5 ms  (31.0 ms/step)
VAE decode:       150.8 ms
```

→ **Denoising이 75%, VAE가 18%, 나머지 7%**.

## 11.4 Step 3 — Step 수에 따른 속도/품질

```python
# step3_step_sweep.py
for num_steps in [4, 8, 20, 50]:
    torch.cuda.synchronize(); t0 = time.time()
    image = pipe(prompt, num_inference_steps=num_steps).images[0]
    torch.cuda.synchronize(); t1 = time.time()
    print(f"steps={num_steps:3d}: {(t1-t0)*1000:.1f} ms")
    image.save(f"step_{num_steps}.png")
```

**예상**:
```
steps=  4:  280 ms (낮은 품질)
steps=  8:  430 ms (보통)
steps= 20:  860 ms (좋음)
steps= 50: 1900 ms (거의 같은 품질)
```

→ **4 step도 동작은 한다** (품질 trade-off).

## 11.5 Step 4 — torch.compile 적용

```python
# step4_compile.py
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")

# Warmup (컴파일 시간 포함, 첫 1~2회는 느림)
for _ in range(3):
    _ = pipe(prompt, num_inference_steps=20).images[0]

# 측정
torch.cuda.synchronize(); t0 = time.time()
image = pipe(prompt, num_inference_steps=20).images[0]
torch.cuda.synchronize(); t1 = time.time()
print(f"With torch.compile: {(t1-t0)*1000:.1f} ms")
```

**예상 가속**: 1.2~1.5배.

## 11.6 자가 점검

1. SD 1.5 추론에서 시간의 가장 큰 비중은 어느 단계인가?
2. Step 수를 4로 줄이면 어떤 trade-off가 발생하는가?
3. `torch.cuda.synchronize()` 가 측정에 왜 필요한가?
4. Warmup을 안 하면 측정이 어떻게 잘못되는가?
5. torch.compile이 가속하는 원리를 한 줄로 설명하라.

---

# 부록 A — 자주 헷갈리는 수식 사전

## A.1 Forward / Reverse

| 수식 | 의미 |
|---|---|
| $x_t = \sqrt{1-\beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon$ | DDPM forward 한 step |
| $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$ | Forward 한 번에 점프 |
| $\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)$ | 누적 alpha |

## A.2 Loss

| 수식 | 의미 |
|---|---|
| $\mathcal{L}_{\text{DDPM}} = \mathbb{E} \| \epsilon - \epsilon_\theta(x_t, t) \|^2$ | 노이즈 예측 손실 |
| $\mathcal{L}_{\text{RF}} = \mathbb{E} \| (x_1 - x_0) - v_\theta(x_t, t) \|^2$ | Velocity 예측 손실 |

## A.3 Sampling

| 수식 | 의미 |
|---|---|
| $x_{t-1} = \mu_\theta(x_t, t) + \sigma_t \epsilon$ | DDPM stochastic |
| $\hat{x}_0 = (x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta) / \sqrt{\bar{\alpha}_t}$ | DDIM 원본 추정 |

## A.4 CFG

| 수식 | 의미 |
|---|---|
| $\epsilon_{\text{guided}} = \epsilon_{\text{uncond}} + s (\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})$ | CFG |

---

# 부록 B — 디퓨전 모델 패밀리 트리

```
                  ┌── DDPM (2020)
                  │     │
                  │     ├── DDIM (2020)
                  │     │     │
                  │     │     └── DPM-Solver, UniPC
                  │     │
                  │     └── Score-based diffusion
                  │
디퓨전 ──────────┤
                  │
                  ├── Latent Diffusion (Stable Diffusion, 2022)
                  │     │
                  │     ├── SD 2.x
                  │     ├── SDXL
                  │     ├── SD3
                  │     │
                  │     └── DiT 기반 ─── PixArt-α
                  │                  └── FLUX.1 / FLUX.2
                  │                         │
                  │                         └── FLUX.2 Klein  ← 이 챌린지
                  │
                  └── Rectified Flow / Flow Matching
                         │
                         ├── SD3 Turbo (1 step)
                         ├── FLUX.1 schnell (1~4 step)
                         └── FLUX.2 Klein (4 step distilled)


Distillation 패밀리:
   DDPM/DDIM
       │
       ├── Progressive Distillation
       ├── Consistency Models
       ├── LCM (Latent Consistency Model)
       ├── SDXL Turbo (ADD)
       └── FLUX.2 Klein
```

---

# 2권 마무리

## 🎯 2권의 결산

이 권을 끝낸 당신은:

✅ 디퓨전 모델이 GAN/VAE를 압도한 본질적 이유를 안다.
✅ DDPM → DDIM → Latent → Rectified Flow → Distillation의 흐름과 각 단계의 트릭을 이해한다.
✅ Stable Diffusion 추론을 수동으로 9단계로 분해해서 짤 수 있다.
✅ FLUX.2 Klein이 4 step distilled, CFG=1.0인 이유와 함의를 안다.
✅ 디퓨전이 LLM의 KV cache를 못 쓰는 이유, 그럼에도 캐싱 가능한 부분을 안다.
✅ 실제 GPU에서 시간 분포를 측정해서 병목을 식별한다.

## 다음 권으로

다음 권 후보:

- **Volume 3 — Transformer 아키텍처 깊이**: 이 챌린지의 백본 (DiT) 을 코드 레벨로.
- **Volume 5 — GPU 시스템**: 백엔드 친화 톤으로 GPU 메모리·연산·정밀도.
- **Volume 7 — 캐싱 이론과 실전**: 이 권 10장에서 본 통찰을 깊이 발전.

다음으로 무엇을 펼치겠는가? 결정은 당신의 몫.
