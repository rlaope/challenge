# 『Q1 교과서 — 제5부 GPU / 커널 / 정밀도 심화편』
> [`textbook_q1.ko.md`](./textbook_q1.ko.md) 의 5부를 **한 권의 책 분량으로 확장한 심화 교재**.
> Q1 챌린지의 진짜 승부처인 **GPU 활용**을 깊게 파고드는 학습서.

---

## 머리말

5부는 다른 부와 다르다.
1~4부가 "**모델을 어떻게 보느냐**"의 시야였다면, 5부는 "**모델을 어떤 기계 위에서 돌리느냐**"의 시야다.

같은 모델, 같은 알고리즘이라도 GPU를 잘 쓰는 사람과 못 쓰는 사람의 차이가 **5~10배**다.
그 격차를 만드는 것이 이 5부의 모든 내용이다.

이 책은 다음 흐름으로 간다:

1. **GPU 자체의 구조** — Hopper, Ada가 어떻게 생겼는지 (Chapter 0).
2. **숫자 표현 — 정밀도** — FP32부터 FP4까지 (Chapter 1~3).
3. **메모리 계층 — IO가 진짜 병목** — HBM, L2, SRAM (Chapter 4).
4. **연산 — Tensor Core와 행렬 곱** — GEMM의 모든 것 (Chapter 5~6).
5. **커널 — FlashAttention** 의 깊은 이해 (Chapter 7~9).
6. **NVIDIA 추론 스택** — TensorRT, cuDNN, CUTLASS, Triton, TransformerEngine (Chapter 10~14).
7. **그래프화 — torch.compile, CUDA Graphs** (Chapter 15~16).
8. **양자화 — Quantization 종합** (Chapter 17~19).
9. **종합 — 이 챌린지에서의 적용 전략** (Chapter 20).

각 장의 구조는 동일하다:
> **🎯 한 줄 정의 → 🎈 쉬운 비유 → 🤔 왜 필요한가 → 😣 옛날엔 무엇이 불편했나 → 💡 어떻게 해결했는가 → 🔬 더 깊이 → 🔥 이 챌린지에서**

---

# 제0장. GPU 아키텍처 입문 — Hopper와 Ada

## 0-1. GPU란 무엇인가? — CPU와의 차이

### 🎯 한 줄 정의

**GPU는 단순한 작업을 동시에 수만 개 처리하는 데 특화된 프로세서.** CPU가 박사급 일꾼 8명이라면, GPU는 초등학생 일꾼 10,000명이다.

### 🎈 쉬운 비유

> 회사가 보고서 1만 부를 복사해야 한다고 하자.
> - **CPU 방식**: 박사 1명이 한 부씩 정성껏 복사. 빠르지만 1만 부면 너무 오래.
> - **GPU 방식**: 알바 1만 명이 동시에 한 부씩. 단순하지만 동시에 끝남.
>
> 이미지 처리, 행렬 곱은 **단순한 작업의 대량 반복**이라 GPU가 압도적.

### 🤔 왜 필요한가

딥러닝의 핵심 연산은 **행렬 곱**.
- 4096×4096 행렬 곱 = 약 680억 번의 곱셈/덧셈.
- CPU로는 하루 종일.
- GPU 하나로 1초 미만.

### 😣 옛날엔 무엇이 불편했나

옛날엔 GPU가 **그래픽 전용** 이었다 (이름 그대로 Graphics Processing Unit).
- 게임, 3D 렌더링.
- 일반 계산은 **shader 트릭** 으로 우회 (GPGPU).
- 매우 복잡, 비효율.

### 💡 어떻게 해결했는가

**CUDA (NVIDIA, 2007)** 의 등장:
- C 언어로 GPU 프로그래밍 가능.
- "**범용 GPU 컴퓨팅**"의 표준.
- 딥러닝 폭발의 기반.

이후:
- **2012 AlexNet** → GPU로 학습 → 딥러닝 시대.
- **2017 Volta (V100)** → Tensor Core 도입.
- **2020 Ampere (A100)** → BF16, sparsity.
- **2022 Hopper (H100)** → FP8, TMA, async.
- **2024 Blackwell (B200)** → FP4, NVLink 5.

### 🔬 GPU의 내부 구조

NVIDIA GPU는 다음 계층:

```
GPU
 └── SM (Streaming Multiprocessor) × N개
      ├── Warp Scheduler × 4
      ├── CUDA Core × 128 (FP32 ALU)
      ├── Tensor Core × 4 (행렬 곱 전용)
      ├── Register File (256 KB)
      └── Shared Memory / L1 Cache (192~256 KB)
```

H100:
- **132개 SM**.
- 각 SM에 **128 CUDA core + 4 Tensor core**.
- 총 16,896 CUDA core.

RTX 4090:
- **128개 SM**.
- 총 16,384 CUDA core.

### 🔑 SM이 GPU의 "코어 단위"

- 하나의 SM이 한 번에 수십~수백 개 thread를 동시 실행.
- thread들은 **warp** (32개 묶음) 단위로 동기.

---

## 0-2. Hopper (H100, SM 90) — 무엇이 새로운가

### 🎯 한 줄 정의

**2022년 NVIDIA의 데이터센터 GPU 아키텍처.** FP8, TMA, Thread Block Cluster 등 LLM/디퓨전 시대의 핵심 기능 도입.

### 🎈 쉬운 비유

> 자동차로 비유:
> - Ampere (A100) = 잘 만들어진 SUV.
> - Hopper (H100) = **터보 + 자율주행 + 새 변속기** 가 추가된 다음 세대.

### 🔬 Hopper의 핵심 기능들

#### (1) FP8 Tensor Core
- E4M3 (forward) + E5M2 (backward) 형식 지원.
- BF16 대비 **2배 빠른 처리량**.
- TransformerEngine으로 자동 활용.

#### (2) Thread Block Cluster
- 여러 SM에 걸친 thread block 그룹.
- 더 큰 단위 협력 가능.
- 예: 큰 attention의 분산 처리.

#### (3) Tensor Memory Accelerator (TMA)
- HBM ↔ SRAM 간 **비동기 메모리 복사** 전용 하드웨어.
- 기존엔 thread가 직접 데이터 옮김 (느림).
- TMA가 백그라운드에서 처리 → 계산과 메모리 IO **오버랩** 가능.

#### (4) Warpgroup MMA (WGMMA)
- 4개 warp(=128 thread)를 묶어 한 번에 큰 행렬 곱.
- FlashAttention-3가 직접 활용.

#### (5) DPX Instructions
- Dynamic Programming 용 명령어 (서열 정렬, 경로 탐색).

#### (6) Distributed Shared Memory
- Cluster 내 SM 간 직접 통신.
- 재료 공유.

### 🔥 이 챌린지에서

H100의 **FP8 + TMA + WGMMA** 의 조합이 FlashAttention-3의 핵심 가속 동력.
**FlashAttention-3 + FP8 + TensorRT** 조합이 가장 큰 단일 가속 카드.

---

## 0-3. Ada Lovelace (RTX 4090, SM 89) — 컨슈머급 강자

### 🎯 한 줄 정의

**2022년 NVIDIA의 컨슈머/워크스테이션 GPU 아키텍처.** Hopper와 같은 세대지만 데이터센터 기능은 일부 빠짐.

### 🔬 Ada vs Hopper

| 기능 | Ada (4090) | Hopper (H100) |
|---|---|---|
| FP8 Tensor Core | ✅ | ✅ |
| TMA | ❌ | ✅ |
| WGMMA | ❌ (제한적) | ✅ |
| Thread Block Cluster | ❌ | ✅ |
| HBM | ❌ (GDDR6X) | ✅ (HBM3) |
| NVLink | ❌ (PCIe만) | ✅ |
| FP64 | 매우 약함 | 보통 |
| 가격 | 약 $1,600 | 약 $30,000+ |

### 🔥 이 챌린지에서

- 4090에서도 FP8은 가능하지만 TMA·WGMMA가 없어 H100만큼 가속 안 됨.
- **메모리 대역폭** 차이가 큼: HBM3 (3TB/s) vs GDDR6X (1TB/s).
- 4090은 **"개발 환경"** 에 적합, H100은 **"프로덕션"** 에 적합.

> 챌린지가 두 GPU를 모두 명시한 이유:
> "어디서 작동해야 하는지에 따라 최적화 전략이 다르다."

---

# 제1부 — 숫자 표현 (정밀도)

## 1장. FP32 (Single Precision) — 표준 부동소수점

### 🎯 한 줄 정의

**32비트 부동소수점.** 1 sign + 8 exponent + 23 mantissa. 거의 모든 컴퓨팅의 기본.

### 🎈 쉬운 비유

> 소수를 **소수점 7자리** 까지 표현 가능한 표준 형식.
> 예: 3.1415926.

### 🤔 왜 기본인가

- 정확도가 충분.
- CPU/GPU 모두 자연스럽게 지원.
- IEEE 754 표준.

### 😣 딥러닝에서 무엇이 불편했나

딥러닝은 **정확도가 약간 떨어져도 괜찮은** 응용:
- Loss surface가 부드러움.
- Gradient의 small noise는 regularization 역할.
- → FP32는 **과스펙**.

게다가:
- **메모리 4배** 더 씀 (FP8 대비).
- **연산 4~8배** 느림 (Tensor Core 활용 시).

### 💡 어떻게 해결했는가

저정밀도(FP16/BF16/FP8)로 옮겨감.
하지만 **gradient나 loss 계산** 같은 일부는 FP32 유지 (mixed precision).

### 🔬 비트 구조

```
S | EEEEEEEE | MMMMMMMMMMMMMMMMMMMMMMM
1   8           23
```

- Range: ±1.18e-38 ~ ±3.4e38.
- Precision: 약 7 decimal digits.

### 🔥 이 챌린지에서

- 거의 안 씀. FP32 추론은 너무 느림.
- 단, 일부 layer (예: LayerNorm)는 FP32 유지가 안전.

---

## 2장. FP16 vs BF16 — 같은 16비트, 다른 철학

### 🎯 한 줄 정의

- **FP16**: 1 sign + 5 exp + 10 mantissa. **정밀도 우선**.
- **BF16**: 1 sign + 8 exp + 7 mantissa. **range 우선**.

### 🎈 쉬운 비유

> 같은 한 페이지에 글을 쓴다고 하자.
> - **FP16**: 작은 글씨로 빽빽이 (디테일 많음, 큰 숫자 표현 어려움).
> - **BF16**: 큰 글씨 듬성듬성 (디테일 적음, 큰 숫자 가능).
>
> 학습에는 큰 숫자(loss, gradient norm)를 다뤄야 해서 BF16이 안전.

### 🤔 왜 두 형식이 다 있는가

**FP16의 등장 (2016, NVIDIA Pascal)**
- "메모리 절반, 속도 2배" 의 첫 시도.
- 그러나 **range가 좁아** gradient underflow 발생.
- → **Loss scaling** (loss × 1024 같은 트릭) 으로 우회.

**BF16의 등장 (2018, Google TPU → NVIDIA Ampere)**
- Loss scaling 없이도 안정.
- FP32와 같은 exponent → underflow 안전.
- 단, mantissa가 적어 **세밀한 정밀도** 약간 떨어짐.

### 😣 옛날엔 무엇이 불편했나

FP16만 있을 때:
- 학습이 자주 NaN으로 폭발.
- Loss scaling 튜닝 필요.
- 큰 모델일수록 더 까다로움.

### 💡 BF16의 우아한 해결

- 비트 수는 같음 (16).
- **Exponent를 FP32와 동일** 하게 → range 같음.
- Mantissa 줄임 → 정밀도만 약간 손해.

학습 안정성이 압도적으로 좋아 거의 모든 LLM이 BF16 사용.

### 🔬 비트 구조 비교

```
FP32: S | EEEEEEEE | MMMMMMMMMMMMMMMMMMMMMMM
BF16: S | EEEEEEEE | MMMMMMM
FP16: S | EEEEE    | MMMMMMMMMM
```

- FP16 range: ±6.55e4 (좁음).
- BF16 range: ±3.4e38 (FP32와 같음).
- FP16 precision: 약 3 decimal digits.
- BF16 precision: 약 2~3 decimal digits.

### 🔥 이 챌린지에서

- **추론** 에선 FP16/BF16 둘 다 안전.
- 입력 분포가 정해져 있으니 underflow 위험 적음.
- BF16이 **수치적으로 더 안전한 기본값**.

---

## 3장. FP8 — 차세대 표준

### 🎯 한 줄 정의

**8비트 부동소수점. 두 변종 존재 — E4M3 (forward용), E5M2 (backward용).** Hopper와 Ada에서 Tensor Core가 직접 지원.

### 🎈 쉬운 비유

> FP16보다 **글씨를 두 배 크게** 써서 같은 페이지에 절반만 들어가게 한 것.
> 정밀도는 떨어지지만 **메모리/대역폭이 절반**.

### 🤔 왜 필요한가

추론에서 **메모리 대역폭** 이 가장 큰 병목.
- 모델 weight를 HBM에서 SRAM으로 옮기는 시간이 곧 전체 시간.
- weight를 **반으로 줄이면** 즉시 2배 빠를 가능성.

### 😣 옛날엔 무엇이 불편했나

INT8 quantization이 먼저 시도됐지만:
- **Outlier 문제**: 한두 개의 큰 값이 정밀도를 망침.
- **Dynamic range 부족**: scale factor 조정이 까다로움.
- 일부 layer는 INT8로 못 함.

### 💡 어떻게 해결했는가

**FP8 (NVIDIA Hopper, 2022)**:
- **부동소수점**이라 dynamic range가 넓음.
- INT8보다 outlier에 강함.
- 두 형식:
  - **E4M3** (4 exp, 3 mantissa): 정밀도 우선, **forward** 용.
  - **E5M2** (5 exp, 2 mantissa): range 우선, **backward (gradient)** 용.

### 🔬 더 깊이 — Per-tensor scaling

FP8의 좁은 range를 보완하는 핵심 트릭:

```python
# 각 텐서마다 scale 보관
scale = max(abs(tensor)) / 448.0  # 448 = E4M3 max
quantized = tensor / scale  # FP8로 변환
# 사용 시
dequantized = quantized * scale
```

- **Per-tensor scale**: 한 텐서당 하나의 scale.
- **Per-channel scale**: 채널마다 scale (더 정확).
- **Delayed scaling**: 이전 step의 scale 사용해서 현재 적용.

### 🔥 이 챌린지에서

- **TransformerEngine** 사용 시 자동으로 적용됨.
- Linear / matmul / attention의 input·output·weight 모두 FP8화 가능.
- **품질 손실은 BF16 대비 1% 미만** (잘 했을 때).
- **속도는 1.5~2배** 가능.

> H100에서 챌린지 4배 가속 목표 중 **이 하나가 1.5~2배** 책임.

---

## 추가: FP4 (Blackwell B200) — 미래

### 🎯 한 줄 정의

**4비트 부동소수점. NVIDIA Blackwell (2024)에서 새로 도입.** 더 극단적인 압축.

### 🎈 쉬운 비유

> 글씨를 정말 큼지막하게 한 페이지에 4글자만 쓰는 셈.
> 메모리 1/8, 속도 4배 가능 (FP32 대비).

### 현재 한계

- 매우 새 기술.
- B200 아직 보급 초기.
- 챌린지(H100/4090) 범위 외지만, "**이게 곧 표준이 된다**"는 트렌드 인식.

---

# 제2부 — 메모리 계층 (IO가 진짜 병목)

## 4장. GPU 메모리 계층 — HBM, L2, SRAM, Register

### 🎯 한 줄 정의

**GPU 안에는 여러 종류의 메모리가 계층적으로 있다.** 위로 갈수록 빠르지만 작고, 아래로 갈수록 느리지만 크다.

### 🎈 쉬운 비유

> 도서관과 책상 비유:
> 1. **Register (책상 위 펜)** — 즉시 접근, 매우 작음.
> 2. **Shared Memory / L1 (책상 위 노트)** — 빠름, 작음.
> 3. **L2 Cache (책상 옆 책꽂이)** — 중간.
> 4. **HBM (도서관 본관)** — 큼, 느림.
> 5. **CPU RAM (옆 건물)** — 더 큼, 더 느림.
> 6. **SSD (창고)** — 거대, 매우 느림.

### 🔬 H100의 실제 수치

| 계층 | 크기 | 대역폭 | 지연 |
|---|---|---|---|
| Register | 256 KB / SM | ~80 TB/s | ~1 cycle |
| Shared Memory / L1 | 256 KB / SM | ~33 TB/s | ~30 cycle |
| L2 Cache | 50 MB | ~7 TB/s | ~200 cycle |
| HBM3 | 80 GB | 3 TB/s | ~500 cycle |
| PCIe Gen5 (CPU) | 시스템 RAM | 64 GB/s | ~수천 cycle |

> 한 cycle = ~0.6 ns.
> Register vs HBM = **500배 차이**.

### 🤔 왜 이게 중요한가

**Memory-bound vs Compute-bound**:
- 작은 행렬 곱: 데이터를 가져오는 게 더 오래 걸림 → memory-bound.
- 큰 행렬 곱: 계산이 더 오래 걸림 → compute-bound.

**Roofline Model**:
```
Performance = min(Compute_peak, Bandwidth × Arithmetic_Intensity)
```

- Arithmetic Intensity = 연산 수 / 메모리 IO 수.
- 낮으면 → bandwidth가 병목.
- 높으면 → compute가 병목.

### 😣 옛날엔 무엇이 불편했나

옛날엔 모든 데이터를 **HBM에서 매번** 읽었다.
- 행렬 곱 한 번에 weight를 여러 번 가져와야 함.
- 메모리 IO가 전체 시간의 80~90%.

### 💡 어떻게 해결했는가 — 데이터 재사용

**Tiling**:
- 큰 행렬을 작은 타일로 나눔.
- 한 타일을 **SRAM에 올려놓고 여러 번 사용**.
- HBM ↔ SRAM 이동 횟수 감소.

이게 **모든 GPU 최적화의 기본 원리**.
- BLAS 라이브러리 (cuBLAS, CUTLASS) 가 이미 구현.
- FlashAttention이 attention에 적용한 케이스.

### 🔥 이 챌린지에서

- 디퓨전 모델은 **작은 batch** 추론 → **memory-bound 경향**.
- 따라서 **메모리 IO 줄이기** 가 가장 큰 가속 동력.
- 구체적으로:
  - **FlashAttention** 으로 attention IO 감소.
  - **FP8** 으로 weight IO 절반.
  - **Layer fusion** 으로 중간 결과를 메모리에 안 씀.

---

## 4-1. HBM (High Bandwidth Memory)

### 🎯 한 줄 정의

**3D로 쌓아올린 DRAM.** GPU와 가까이 붙여서 매우 높은 대역폭 제공.

### 🎈 쉬운 비유

> 일반 DRAM = 옆 책상.
> HBM = 책상에 직접 붙은 책꽂이.
> **데이터 길이가 짧아져서 더 빨리 옮길 수 있음.**

### 세대

- HBM2 (P100, V100): ~700 GB/s.
- HBM2e (A100): ~2 TB/s.
- HBM3 (H100): ~3 TB/s.
- HBM3e (H200, B200): ~5 TB/s.

### 🔥 RTX 4090과의 차이

- 4090: GDDR6X, ~1 TB/s.
- H100: HBM3, ~3 TB/s.
- **3배 차이** → 같은 모델도 H100이 3배 가까이 빠를 수 있음.

---

## 4-2. SRAM / Shared Memory — 가장 빠른 작은 공간

### 🎯 한 줄 정의

**SM 안에 있는 작은 고속 메모리.** 같은 thread block의 thread들이 공유.

### 🎈 쉬운 비유

> 한 팀(thread block)이 함께 쓰는 화이트보드.
> 빠르게 적고 지울 수 있음.

### 사용 패턴

```cuda
__shared__ float tile[16][16];
// 같은 block의 thread들이 협력해서 채움
tile[ty][tx] = global_data[...];
__syncthreads();
// 이제 빠르게 사용
```

### 🔥 FlashAttention의 핵심

- Q, K, V의 작은 chunk를 SRAM에 올림.
- **거기서 모든 attention 계산을 끝냄**.
- 중간 결과를 HBM에 안 씀.
- → 메모리 IO 폭감.

---

# 제3부 — 연산 (Tensor Core와 GEMM)

## 5장. CUDA Core vs Tensor Core

### 🎯 한 줄 정의

- **CUDA Core**: 일반 부동소수점 ALU. 한 번에 한 곱셈.
- **Tensor Core**: 작은 행렬 곱을 한 번에 처리하는 전용 유닛.

### 🎈 쉬운 비유

> 종이접기 공장:
> - **CUDA Core**: 일꾼이 한 번에 종이 한 장씩 접기.
> - **Tensor Core**: 기계가 4×4 종이를 한 번에 접기.
>
> 행렬 곱은 본질적으로 **수많은 작은 행렬 곱**의 누적 → Tensor Core가 압도적.

### 🔬 더 깊이 — Tensor Core의 동작

한 번의 Tensor Core 명령:
```
D = A × B + C
```
- A, B, C, D는 작은 행렬 (예: 16×16, 16×8 등).
- **하나의 명령으로 256개 곱셈 + 256개 덧셈**.

세대별:
- Volta: 4×4×4 FP16.
- Ampere: 8×8×4 BF16, TF32.
- Hopper: 16×8×16 FP8 (WGMMA).

### 🤔 왜 필요한가

- 일반 ALU로 256 곱셈 = 256 cycle.
- Tensor Core로 같은 일 = 1 cycle.
- **256배 빠른 셈**.

### 😣 옛날엔 무엇이 불편했나

Volta (2017) 이전엔 행렬 곱을 일반 ALU로 처리.
- 딥러닝 가속이 한계에 부딪힘.
- 학습 시간 폭증.

### 💡 Tensor Core의 등장

- Volta V100 (2017) — 첫 도입.
- 이후 모든 NVIDIA GPU에 포함.
- 사용 패턴:
  - cuBLAS / cuDNN이 자동으로 호출.
  - 직접 짤 때는 PTX 또는 CUTLASS / Triton.

### 🔥 이 챌린지에서

- 행렬 곱 (Linear, Attention의 QK·SV) 은 **Tensor Core 필수**.
- FP8 Tensor Core 활용이 H100에서의 큰 카드.

---

## 6장. GEMM (General Matrix Multiply) — 모든 딥러닝의 기초

### 🎯 한 줄 정의

**General Matrix-Matrix Multiplication.** `C = A × B + C`. 딥러닝 연산의 90%가 사실상 GEMM.

### 🎈 쉬운 비유

> 곱셈 구구단을 거대한 표로 만든 것.

### 🤔 왜 핵심인가

**Linear layer**:
```
output = input @ weight + bias  ← GEMM
```

**Attention**:
- `Q = X @ W_q` ← GEMM
- `Score = Q @ K^T` ← GEMM
- `Out = Softmax(Score) @ V` ← GEMM

**Convolution**:
- im2col 변환 후 GEMM.

**MLP, FFN**: GEMM.

### 😣 옛날엔 무엇이 불편했나

GEMM 구현은 매우 어렵다:
- 메모리 계층 활용.
- Tensor Core 호출.
- 다양한 행렬 모양 (Tall, Wide, Square).

직접 짜면 학술 라이브러리 수준 성능.

### 💡 어떻게 해결했는가

**cuBLAS** (NVIDIA, 1990년대~):
- 표준 BLAS API의 GPU 구현.
- 자동으로 최적 커널 선택.
- 거의 모든 딥러닝 프레임워크가 사용.

**CUTLASS** (NVIDIA, 2017~):
- C++ template 라이브러리.
- 사용자가 GEMM의 부품을 조합해서 **커스텀 변종** 만들기.
- FlashAttention, fused kernel 등의 빌딩 블록.

### 🔬 GEMM 최적화의 단계 (학습 순서)

1. **Naive triple loop** (느림).
2. **Loop tiling** (캐시 활용).
3. **Register blocking** (레지스터 재사용).
4. **Vectorization** (SIMD/Tensor Core).
5. **Software pipelining** (메모리 IO와 계산 overlap).
6. **Async copy + double buffering** (Hopper의 TMA).

각 단계마다 2~3배 빨라짐.

### 🔥 이 챌린지에서

- 직접 GEMM 짤 일은 거의 없음.
- 그러나 **어떤 라이브러리를 쓸지** 결정이 중요:
  - PyTorch eager → cuBLAS (기본).
  - TensorRT → 더 최적화된 커널.
  - Custom (Triton/CUTLASS) → 특수 형태.

---

# 제4부 — FlashAttention의 깊은 이해

## 7장. 표준 Attention의 문제

### 🎯 한 줄 정의

**표준 Self-Attention은 메모리 사용량이 시퀀스 길이의 제곱(O(N²)).** 긴 시퀀스에서 메모리 부족.

### 🎈 쉬운 비유

> 4096명이 모인 회의에서 모두가 서로의 이름을 적은 명단을 받아야 한다면:
> - 4096 × 4096 = 약 1700만 줄의 명단.
> - 종이가 모자란다.

### 🔬 표준 Attention 의 흐름

```python
S = Q @ K.T          # (N, N) 행렬 — 메모리 폭주
P = softmax(S)       # (N, N)
O = P @ V            # (N, d)
```

문제:
- S와 P를 **HBM에 저장** 해야 함.
- N=4096 → 16MB (FP32 기준), 4MB (FP16 기준) per head.
- 24 head, 25 block → 수 GB.
- 계산 자체는 빠른데 **메모리 IO가 병목**.

### 😣 옛날엔 무엇이 불편했나

긴 시퀀스 (LLM, 디퓨전) 에서:
- OOM (Out of Memory).
- 또는 OOM 안 나도 **메모리 IO로 매우 느림**.

### 💡 부분적 해결 — Memory-efficient attention

PyTorch가 자체적으로 attention을 chunk 단위로 처리하기 시작 (2021).
- 메모리는 줄였지만 속도는 느렸음.

### 🔥 더 근본적 해결이 필요했음

→ FlashAttention의 등장.

---

## 8장. FlashAttention 1 — 첫 혁명

### 🎯 한 줄 정의

**Attention을 GPU 메모리 계층에 맞게 재구성한 알고리즘.** 중간 행렬을 HBM에 안 쓰고 SRAM에서 처리.

### 🎈 쉬운 비유

> 4096명 회의를 작은 회의실 (SRAM) 에서 16명씩 묶어 처리.
> - 16명끼리만 명단 작성.
> - 작성 후 **누가 누구에게 연관된지 결과만** 큰 노트(HBM) 에 기록.
> - 중간 명단은 버림.

### 🔬 핵심 아이디어 — Online Softmax

표준 softmax는 한 번에 전체 행 보고 계산.
FlashAttention은 **청크 단위로 누적 계산**.

```python
# 의사 코드
for block_K, block_V in K, V:  # K, V를 청크로 나눔
    for block_Q in Q:  # Q도 청크로
        # SRAM에 이 청크들 올림
        s = block_Q @ block_K.T
        # online softmax로 max, sum 누적 갱신
        m_new = max(m_old, max(s))
        l_new = exp(m_old - m_new) * l_old + sum(exp(s - m_new))
        # 출력 누적
        o = exp(m_old - m_new) * o + exp(s - m_new) @ block_V
    # 최종 정규화
    o = o / l_new
```

핵심:
- **S, P를 HBM에 절대 안 씀**.
- 각 청크가 SRAM에 들어가는 크기.
- max, sum을 점진적으로 갱신.

### 결과

- 메모리: O(N²) → O(N).
- 속도: **2~4배** (긴 시퀀스에서).
- 같은 결과 보장 (수치적 동치).

### 🔥 이 챌린지에서

- FLUX의 attention에 적용 가능.
- 보통 PyTorch의 `scaled_dot_product_attention` 이 자동으로 FlashAttention 호출.

---

## 9장. FlashAttention 3 — Hopper 전용 가속

### 🎯 한 줄 정의

**Hopper (H100) 의 새 하드웨어 (TMA, WGMMA, FP8) 를 100% 활용한 attention 커널.**

### 🎈 쉬운 비유

> FlashAttention 1이 새 운영체제라면, FA3는 그 운영체제를 새 CPU에 맞게 재컴파일한 것.

### 🔬 새로운 점

#### (1) Warpgroup MMA 활용
- 4 warp = 128 thread를 묶어 큰 행렬 곱 한 번에.
- WGMMA 명령어가 비동기 실행 → 다른 작업과 overlap.

#### (2) TMA (Tensor Memory Accelerator)
- HBM ↔ SRAM 데이터 이동을 비동기로.
- 계산하는 동안 다음 청크를 자동 로딩.
- **Memory IO와 compute 완전 overlap**.

#### (3) FP8 지원
- Q, K, V를 FP8로.
- 더 큰 청크가 SRAM에 들어감.
- 대역폭 절반.

#### (4) Producer-Consumer 분할
- 한 warp는 데이터 로딩 (producer).
- 다른 warp는 계산 (consumer).
- 완전 비동기.

### 결과

- FlashAttention 2 대비 **1.5~2배**.
- BF16 대비 H100에서 75% peak FLOPS 활용 (기존 35%에서 두 배).

### 🔥 이 챌린지에서

- H100을 쓰면 **FA3 적용이 거의 필수**.
- PyTorch 2.4+ 또는 직접 `flash-attn` 라이브러리.
- 디퓨전의 attention들에 적용 → 큰 가속.

---

# 제5부 — NVIDIA 추론 스택

## 10장. CUDA — 모든 것의 기반

### 🎯 한 줄 정의

**NVIDIA GPU를 프로그래밍하기 위한 C/C++ 확장 + 컴파일러 + 런타임.**

### 🎈 쉬운 비유

> GPU 운영체제의 시스템 콜 같은 것.
> 모든 상위 라이브러리가 CUDA 위에 만들어짐.

### 🤔 왜 필요한가

GPU 코드는 CPU 코드와 다름:
- 수천 개 thread 동시 실행.
- 메모리 계층 명시적 관리.
- 동기화 (`__syncthreads()`).

이걸 추상화한 것이 CUDA.

### 😣 옛날엔 무엇이 불편했나

CUDA 이전 GPGPU는 OpenGL shader로 우회.
- 수치 계산을 픽셀 색깔로 표현 (!).
- 매우 비효율, 학습 곡선 가파름.

### 💡 CUDA의 모델

- **Thread**: 가장 작은 실행 단위.
- **Block**: thread들의 그룹 (보통 256~1024).
- **Grid**: block들의 집합.
- **Warp**: 32 thread (실행 단위).

```cuda
__global__ void add(float* a, float* b, float* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

// 호출
add<<<numBlocks, threadsPerBlock>>>(a, b, c);
```

### 🔥 이 챌린지에서

- 직접 CUDA 짤 일은 없을 가능성 높음.
- 그러나 **CUDA 모델 이해** 가 위 라이브러리 이해에 도움.

---

## 11장. cuDNN — 딥러닝 표준 라이브러리

### 🎯 한 줄 정의

**Convolution, RNN, Attention 등 딥러닝 표준 연산을 GPU에서 빠르게 실행하는 NVIDIA 공식 라이브러리.**

### 🎈 쉬운 비유

> 표준 부품 가게.
> "Conv2D 주세요" → 가장 빠른 구현 제공.

### 🤔 왜 필요한가

같은 Conv도 입력 모양에 따라 최적 알고리즘이 다름:
- Direct convolution.
- im2col + GEMM.
- Winograd.
- FFT.

cuDNN이 자동 선택.

### 🔥 이 챌린지에서

- VAE의 Conv 연산은 cuDNN 사용.
- 자동으로 적용되므로 직접 호출할 일은 적음.

---

## 12장. CUTLASS — GEMM 빌딩 블록

### 🎯 한 줄 정의

**C++ template 기반 GEMM/Convolution 빌딩 블록 라이브러리.** 사용자가 부품을 조합해 커스텀 커널 작성.

### 🎈 쉬운 비유

> LEGO 세트.
> 표준 부품을 가져다 자기만의 모델을 만든다.

### 🤔 왜 필요한가

cuBLAS는 표준 GEMM만 빠르게 함.
**Fused kernel** (예: GEMM + activation + bias 한 번에) 이나 **이상한 모양** 에는 약함.

CUTLASS로 직접 짜면:
- Fused 가능.
- 새 데이터 타입 (FP8) 즉시 지원.
- FlashAttention 같은 커스텀 알고리즘 가능.

### 🔥 이 챌린지에서

- 직접 CUTLASS 짤 가능성 낮음.
- 그러나 FlashAttention, TransformerEngine 등이 내부적으로 CUTLASS 사용.

---

## 13장. Triton (OpenAI) — 쉬운 GPU 커널 작성

### 🎯 한 줄 정의

**Python 스타일로 GPU 커널을 작성할 수 있는 DSL.** CUDA보다 훨씬 쉽고, cuBLAS 수준 성능 가능.

### 🎈 쉬운 비유

> 영어로만 쓰던 책을 **한국어로 번역** 한 것.
> 진입 장벽 폭감.

### 🤔 왜 필요한가

CUDA는 어렵다:
- 메모리 계층 직접 관리.
- Tensor Core 수동 호출.
- 디버깅 어려움.

Triton (Tillet et al., 2019, OpenAI 인수):
- Python 데코레이터로 커널 정의.
- 메모리 계층은 **block 단위**로만 추상화.
- 자동으로 Tensor Core 활용.

### 💡 예제

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

CUDA로 같은 코드 짜면 3배 길고 복잡.

### 🔥 이 챌린지에서

- FlashAttention 2 가 Triton 구현 있음.
- 새로운 커널 작성 시 Triton이 첫 선택.
- PyTorch 2.0의 `torch.compile` Inductor 백엔드가 Triton 코드 생성.

---

## 14장. TensorRT / TensorRT-LLM — 추론 컴파일러

### 🎯 한 줄 정의

**모델을 받아서 GPU에 최적화된 단일 엔진으로 컴파일하는 NVIDIA의 추론 프레임워크.** PyTorch 추론 대비 2~10배 가속.

### 🎈 쉬운 비유

> 자동차 일반 정비 vs **레이싱 튜닝**.
> TensorRT가 레이싱 튜닝.
> 모든 부품을 그 차에 맞게 다시 깎아냄.

### 🔬 무엇을 하는가

#### (1) Layer Fusion
- 여러 작은 연산을 하나로 합침.
- 예: `Conv → BatchNorm → ReLU` → 하나의 fused kernel.
- **메모리 IO 감소**.

#### (2) Kernel Selection
- 같은 연산도 입력 모양에 따라 최적 커널 선택.
- 빌드 시 벤치마크해서 결정.

#### (3) Precision Calibration
- FP32 → FP16/INT8/FP8 변환.
- Calibration data로 정밀도 손실 최소화.

#### (4) Memory Optimization
- 텐서 재사용.
- Workspace 최적화.

### 🤔 왜 필요한가

PyTorch eager는:
- Python 오버헤드.
- 일반화된 커널.
- 정밀도 자동 변환 안 함.

서비스에는 부적합.

### 😣 옛날엔 무엇이 불편했나

추론 모델을 직접 최적화하려면:
- ONNX 변환.
- 커스텀 커널 작성.
- 매번 모델 바뀔 때마다 반복.

### 💡 TensorRT의 워크플로우

1. PyTorch 모델 → ONNX export.
2. TensorRT가 ONNX 받아 분석.
3. `trtexec` 또는 Python API로 엔진 빌드.
4. 빌드된 엔진 (`.plan`) 파일 저장.
5. Python/C++에서 엔진 로드 후 추론.

### 🔬 TensorRT-LLM (LLM 특화)

- KV cache 관리.
- Continuous batching.
- Tensor parallelism.
- LLM 패턴에 특화된 최적화.

### 🔥 이 챌린지에서 — 핵심 카드

NVIDIA가 만든 **디퓨전 데모**:
- `NVIDIA/TensorRT/demo/Diffusion`
- Stable Diffusion / SDXL의 TensorRT 변환 예제.
- FLUX에도 응용 가능 (커뮤니티 작업 활발).

> 단순 "TensorRT로 빌드" 만으로 **2~3배** 가속 가능성.

---

## 15장. TransformerEngine — FP8 자동 관리

### 🎯 한 줄 정의

**Hopper의 FP8 Tensor Core를 자동으로 활용하게 해주는 NVIDIA 라이브러리.** PyTorch 모델의 layer를 FP8 버전으로 교체.

### 🎈 쉬운 비유

> 자동 변속기.
> 운전자가 RPM 보고 변속할 필요 없이, 차가 알아서 적절한 기어로.
>
> TransformerEngine이 FP8/BF16 사이를 자동으로 전환.

### 🔬 핵심 기능

#### (1) FP8 Linear, LayerNorm, Attention 등
- PyTorch nn.Linear → te.Linear.
- 내부에서 FP8 사용.

#### (2) Delayed Scaling
- 매 forward에서 scale 통계 수집.
- 다음 forward에서 그 scale 사용.
- 정확도 유지.

#### (3) Mixed Precision Recipe
- 어떤 layer를 FP8로, 어떤 걸 BF16으로 유지할지 자동.

### 💡 사용 예

```python
import transformer_engine.pytorch as te

# 그냥 Linear → FP8 Linear
linear = te.Linear(in_features, out_features)

# Mixed precision context
with te.fp8_autocast(enabled=True):
    output = linear(input)
```

### 🔥 이 챌린지에서

- FLUX의 Linear / Attention을 FP8로 교체 → 큰 가속.
- 단, 품질 검증 필수.
- 챌린지가 추천하는 NVIDIA 추론 스택의 핵심 도구 중 하나.

---

# 제6부 — 그래프화 (torch.compile, CUDA Graphs)

## 16장. torch.compile — 자동 그래프 컴파일

### 🎯 한 줄 정의

**PyTorch 2.0+ 의 컴파일러.** 모델 코드를 자동으로 그래프화 → 컴파일 → 최적화.

### 🎈 쉬운 비유

> 통역사 vs 번역가:
> - **eager mode** = 통역사 (한 줄씩 즉석 통역).
> - **compile** = 번역가 (전체를 보고 책으로 출판).
>
> 책이 더 효율적이지만 출판 시간 필요.

### 🔬 내부 구조

```
PyTorch code
    ↓ TorchDynamo (Python bytecode 캡처)
FX Graph
    ↓ AOTAutograd (forward/backward 분리)
Optimized Graph
    ↓ Inductor (백엔드)
Triton / C++ Kernel Code
    ↓ 컴파일
Optimized Binary
```

### 사용법

```python
model = MyModel()
compiled_model = torch.compile(model, mode="reduce-overhead")

# 처음 호출 시 컴파일 (느림)
output = compiled_model(input)
# 두 번째부터 빠름
output = compiled_model(input2)
```

### 모드들

- `default`: 일반 (적당한 컴파일).
- `reduce-overhead`: CUDA Graph 활용 (작은 batch에 좋음).
- `max-autotune`: 최대 최적화 (컴파일 매우 느림).

### 🤔 왜 필요한가

PyTorch eager는:
- Python ↔ CUDA 왕복.
- 작은 batch에서 launch overhead 큼.
- 매 layer마다 별도 커널.

torch.compile:
- Operator fusion 자동.
- CUDA Graph 자동 적용.
- Triton 코드 생성.

### 🔥 이 챌린지에서

- FLUX 모델에 한 줄로 적용 가능.
- **즉시 1.3~1.5배** 가속 (보통).
- 단, **dynamic shape** 시 recompile 자주 → bucket으로 묶거나 `dynamic=True`.

---

## 17장. CUDA Graphs — 호출 오버헤드 제로

### 🎯 한 줄 정의

**일련의 GPU 명령을 "그래프"로 묶어서 한 번에 launch하는 메커니즘.** Python 오버헤드 거의 0.

### 🎈 쉬운 비유

> 식당 주문:
> - **일반 launch**: 한 메뉴씩 주문하고 답변 받기.
> - **CUDA Graph**: 메뉴 전체를 종이에 적어 한 번에 전달.

### 🔬 동작 원리

1. **Capture**: 한 번의 실행을 "녹화".
2. **Instantiate**: 녹화된 그래프를 GPU 명령 시퀀스로 변환.
3. **Launch**: 그래프 통째로 GPU에 던짐.

### 사용 패턴

```python
# Capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    output = model(static_input)

# Replay (오버헤드 거의 0)
for i in range(N):
    static_input.copy_(real_input)
    g.replay()
    real_output = output.clone()
```

### 🔬 제약

- **Static shape** 필요.
- 입력 텐서를 **재사용** (메모리 주소 같아야).
- 동적 분기 어려움.

### 🤔 왜 필요한가

작은 batch / 짧은 forward 에서:
- 한 layer의 launch overhead가 ~10μs.
- 100 layer = 1ms.
- 모델 본 계산이 5ms면 **20% 손실**.

CUDA Graph로 **이 오버헤드 제거**.

### 🔥 이 챌린지에서

- 디퓨전 4 step은 같은 forward를 4번 → CUDA Graph 완벽.
- batch=1 추론에서 특히 효과 큼.
- `torch.compile(mode="reduce-overhead")` 가 자동 적용.

---

# 제7부 — 양자화 종합

## 18장. Quantization 기초

### 🎯 한 줄 정의

**FP16/BF16의 가중치/활성값을 더 적은 비트(INT8, INT4 등)로 표현해서 메모리/계산을 줄이는 기법.**

### 🎈 쉬운 비유

> 음원 압축 비유 다시:
> - WAV (32비트) → MP3 (압축) → AAC → Opus.
> - 같은 곡, 점점 작아지는 파일, 듣기엔 거의 같음.

### 🔬 핵심 수식

**Symmetric Quantization (Linear)**:
```
quantized = round(value / scale)
dequantized = quantized × scale
```

`scale = max(abs(value)) / 127` (INT8 기준).

**Asymmetric** (zero-point 사용):
```
quantized = round((value - zero_point) / scale)
```

### 종류 분류

#### Post-Training Quantization (PTQ)
- 학습 끝난 모델에 양자화 적용.
- 빠름.
- 정확도 약간 손해.

#### Quantization-Aware Training (QAT)
- 양자화를 시뮬레이션하며 학습.
- 정확도 좋음.
- 학습 비용 큼.

### 적용 범위

#### Weight-only
- 가중치만 양자화.
- 활성값은 FP16 유지.
- **메모리 절약 위주** (계산은 빠르지 않을 수 있음).

#### Weight + Activation (W8A8 등)
- 둘 다 양자화.
- **계산도 빠름** (INT8 Tensor Core).
- 어려움 (activation의 outlier).

---

## 19장. SmoothQuant, AWQ, GPTQ — 실전 quantization 기법들

### 🎯 한 줄 정의

각각 LLM/디퓨전을 INT8/INT4로 양자화하면서 정확도를 유지하는 영리한 기법들.

### 19-1. SmoothQuant

#### 문제
- Activation에는 **outlier 채널** 이 있음.
- 그 채널 때문에 scale이 너무 커져 다른 값들의 정밀도 손해.

#### 해결
- Outlier를 **weight로 옮김**.
- Activation의 dynamic range 줄임.
- Mathematically equivalent.

```
Original: Y = X @ W
Smoothed: Y = (X / s) @ (s × W)
         = X' @ W'
```

`s` 를 적절히 선택해서 **X' 의 outlier 감소**, **W' 는 outlier 여유 있음**.

### 19-2. AWQ (Activation-aware Weight Quantization)

#### 핵심 통찰
- 모든 weight가 동등하게 중요한 게 아님.
- 큰 activation을 받는 weight는 더 보존해야 함.

#### 해결
- Activation의 magnitude 보고 **중요한 weight 채널 보호**.
- INT4 양자화 가능.

### 19-3. GPTQ

#### 핵심 통찰
- 양자화 오차를 **다음 weight에서 보상**.

#### 해결
- 한 column씩 양자화.
- 각 column의 오차를 다음 column에 더해서 보정.
- Hessian 기반 최적화.

### 🔥 이 챌린지에서 — 디퓨전 양자화

LLM과 다른 점:
- 디퓨전은 **batch 작음** → 메모리보다 **연산량** 이 더 중요.
- 그래서 W4A16 (weight INT4) 보다 **W8A8 (둘 다 FP8)** 가 더 직접적인 가속.

추천 조합:
- FLUX의 Linear → FP8 (TransformerEngine).
- VAE → BF16 유지 (정밀도 중요).
- LayerNorm → FP32 유지 (수치 안정성).

---

# 제8부 — 종합

## 20장. 이 챌린지에서의 GPU 최적화 전략

### 통합된 액션 플랜

#### 가속 카드 우선순위 (효과 큰 순)

```
[1] FlashAttention 3 적용
    └ H100에서 attention 1.5~2배
    └ 적용 비용: 낮음 (PyTorch 2.4+ 또는 flash-attn)

[2] FP8 quantization (TransformerEngine)
    └ Linear/MatMul 1.5~2배
    └ 적용 비용: 중간 (calibration 필요)

[3] torch.compile + CUDA Graph
    └ Python overhead 제거 1.2~1.4배
    └ 적용 비용: 낮음

[4] TensorRT 컴파일
    └ Layer fusion + kernel selection 1.5~2배
    └ 적용 비용: 높음 (ONNX 변환, 디버깅)

[5] Step-wise feature caching (Learning-to-Cache)
    └ Forward pass의 일부 생략 1.3~2배
    └ 적용 비용: 중간 (학습 또는 휴리스틱)

[6] Reference K/V caching (이 챌린지의 핵심 통찰)
    └ 아바타 토큰의 K/V 재사용 1.2~1.5배
    └ 적용 비용: 중간 (코드 수정 필요)

[7] VAE caching (아바타)
    └ 매 클릭에서 VAE encode 생략
    └ 적용 비용: 낮음

[8] Token pruning / merging
    └ Reference token 압축 1.1~1.3배
    └ 적용 비용: 중간 (품질 검증 필수)
```

조합 시 **곱셈 효과** (단, overlap 있음):
- 보수적 추정: 1.5 × 1.5 × 1.2 = **2.7배**.
- 적극적 추정: 2 × 2 × 1.4 × 1.5 = **8.4배**.

> **목표 4배는 충분히 달성 가능**.

### 측정 순서 (실험 계획)

1. **Baseline 측정** (현재 1.4초 재현).
2. **FlashAttention 3 단독** (가장 안전한 가속).
3. **+ torch.compile** (overhead 제거).
4. **+ FP8** (품질 검증 동반).
5. **+ Cache 전략** (가장 큰 통찰).
6. **+ TensorRT 통합** (마지막에 한 번).

각 단계마다:
- Latency (p50, p95) 측정.
- 품질 (LPIPS, CLIP) 측정.
- Regression 없으면 다음 단계.

### 위험 신호

- **"한 번에 다 하기"** → 어디서 망가졌는지 추적 불가.
- **Warmup 없이 측정** → 잘못된 결과.
- **품질 측정 안 하기** → 빠르지만 결과 망가짐.

---

## 맺음말

5부의 모든 내용은 결국 **하나의 큰 그림**을 위한 것이다:

> "이 모델을 이 GPU에서 가능한 최대한 빠르게 돌리고 싶다."

이 목표를 위해:
- **GPU 구조** 를 알고,
- **메모리 IO** 가 병목인 곳을 찾고,
- **연산 단위** 를 Tensor Core로 옮기고,
- **그래프화** 로 오버헤드를 없애고,
- **정밀도** 를 줄여 메모리/속도를 동시에 잡는다.

각 카드를 따로 보면 작아 보이지만, **조합**할 때 폭발적이다.

이 챌린지의 4배 가속은 마법이 아니다.
**적절한 조합과 측정의 결과**일 뿐이다.

> 이제 GPU를 손에 들었다.
> 그 다음은 **모델 안의 어떤 패턴**이 이 카드들과 맞아떨어지는지 찾는 일이다.
> 그건 1~4부의 내용이다.
>
> 5부는 도구. 1~4부는 통찰. **둘이 만나는 자리에 답이 있다.**

---

## 참고 자료

- [`textbook_q1.ko.md`](./textbook_q1.ko.md) — 본편 (1~4부).
- [`pipeline_explained.ko.md`](./pipeline_explained.ko.md) — 파이프라인 그림.
- [`q1_ready.md`](./q1_ready.md) — 체크리스트.

### 추천 외부 자료

- **NVIDIA H100 White Paper**: <https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet>
- **CUDA C++ Programming Guide**: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- **CUTLASS**: <https://github.com/NVIDIA/cutlass>
- **Triton**: <https://triton-lang.org/>
- **TensorRT**: <https://developer.nvidia.com/tensorrt>
- **TransformerEngine**: <https://github.com/NVIDIA/TransformerEngine>
- **FlashAttention 3 paper**: <https://arxiv.org/abs/2407.08608>
- **FlashAttention repo**: <https://github.com/Dao-AILab/flash-attention>
- **PyTorch torch.compile docs**: <https://pytorch.org/docs/stable/torch.compiler.html>

### 추천 학습 순서

1. **CUDA 기초** — Programming Massively Parallel Processors (책).
2. **FlashAttention paper 1 → 2 → 3** — 진화 흐름 이해.
3. **Triton 튜토리얼** — 직접 커널 짜보기.
4. **TensorRT Diffusion 데모 돌려보기** — 실전 감각.
5. **TransformerEngine 예제** — FP8 적용 감각.
