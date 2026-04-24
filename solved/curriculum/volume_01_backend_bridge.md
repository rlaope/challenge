# Volume 1 — 백엔드 엔지니어를 위한 AI 추론 입문

> **이 권의 목적:**
> AI 추론 영역의 **어휘·사고방식·기본 도구**를 백엔드 개발자의 익숙한 개념 위에 얹어, 2권부터 시작될 본격적인 기술 학습(디퓨전·Transformer·GPU)을 받을 준비를 끝낸다.
>
> **이 권을 다 읽으면 할 수 있는 것:**
> 1. AI 추론 엔지니어 직무 설명을 읽고 무엇을 하는 일인지 1분 안에 설명할 수 있다.
> 2. 행렬·텐서·신경망의 동작을 코드 한 줄 한 줄 따라 읽을 수 있다.
> 3. PyTorch로 사전학습 모델을 다운로드하여 추론 한 번을 실제로 실행한다.
> 4. AI 추론 서비스가 일반 REST API와 무엇이 다른지 5가지를 짚어낼 수 있다.
> 5. 다음 권(2~11권) 중 자기 상황에 맞는 학습 순서를 직접 결정한다.

> **분량:** 약 1,800줄 / 정독+실습 시 6~10시간.

---

## 목차

```
[0]   시작하기 전에 — 이 권의 사용법
[1장] 왜 지금 AI 추론 엔지니어인가 — 시장과 직무
[2장] 백엔드 vs AI 추론 — 비슷한 점, 다른 점
[3장] 어휘 변환표 — 백엔드 ↔ AI/ML
[4장] 행렬·텐서 완전 정복 — DB row/column이 익숙하다면 쉽다
[5장] 신경망의 본질 — "거대한 합성 함수다"
[6장] PyTorch 30분 입문 — Flask 만들 줄 알면 친숙하다
[7장] 첫 모델 추론 — Hugging Face로 한 시간 만에 동작 확인
[8장] AI 추론 서비스 vs 일반 REST API — 무엇이 다른가
[9장] 모델의 라이프사이클 — 학습/배포/모니터링
[10장] 백엔드 엔지니어가 가진 강점 — 시스템 사고는 강한 무기
[11장] 학습 경로 결정 — 다음 권 선택 가이드
[부록 A] 환경 셋업 — Python, PyTorch, CUDA
[부록 B] 모호한 표현 사전 — 책에서 피한 표현 목록
```

---

# 0장. 시작하기 전에

## 🎯 목적

이 권을 **어떻게 읽을 것인가**를 정한다. 학습 효율은 첫 30분이 결정한다.

## 0.1 이 책의 약속

이 책은 다음 세 가지를 약속한다.

**약속 1 — 모호한 표현을 쓰지 않는다.**
"~할 수도 있다", "~정도이다", "보통은 ~하다" 같은 표현을 최소화한다. 사실은 사실로, 추정은 추정임을 명시한다.

**약속 2 — 모든 코드는 그대로 복붙해서 동작한다.**
의사코드(pseudocode)와 실행 가능 코드를 명확히 구분한다. 의사코드는 `# pseudocode` 주석으로 표시한다.

**약속 3 — 매 장 끝에 자가 점검 질문이 있다.**
이 질문 5개에 입으로 답할 수 있으면 그 장을 통과한 것이다.

## 0.2 사전 준비물

| 항목 | 상태 확인 명령 | 없으면 |
|---|---|---|
| Python 3.10+ | `python3 --version` | [부록 A](#부록-a--환경-셋업--python-pytorch-cuda) |
| pip 또는 uv | `pip --version` | 위와 동일 |
| Git | `git --version` | OS 기본 패키지 |
| 텍스트 에디터 | VSCode 권장 | <https://code.visualstudio.com> |
| GPU (선택) | `nvidia-smi` | 없어도 6장까지는 진행 가능 |

GPU가 없어도 1권은 끝까지 학습할 수 있다. CPU만으로 동작하는 코드 예제 위주로 구성했다.

## 0.3 학습 흐름도

```
┌──────────────────────────────────────────────────────────┐
│                     1권 학습 흐름도                        │
└──────────────────────────────────────────────────────────┘

[1장] 왜 이 길로 가는가              ← 동기 부여
   │
   ▼
[2장] 비슷한 점부터 발견              ← 자신감 회복
   │
   ▼
[3장] 어휘 매핑                     ← 진입 장벽 해체
   │
   ▼
[4장] 행렬·텐서  ───┐
                  │ 기초
[5장] 신경망 ────┘
   │
   ▼
[6장] PyTorch                      ← 도구
   │
   ▼
[7장] 실제 추론                     ← 손으로 체험
   │
   ▼
[8장] 서비스 차이 ──┐
                  │ 시스템 시야
[9장] 라이프사이클 ──┤
                  │
[10장] 강점 정리 ──┘
   │
   ▼
[11장] 다음 권 선택                 ← 길 정하기
```

---

# 1장. 왜 지금 AI 추론 엔지니어인가

## 🎯 이 장의 목적

**왜 이 길을 가는가**의 답을 명확히 한다. 동기 없는 학습은 3개월을 못 간다.

## 1.1 개요

AI 추론 엔지니어(AI Inference Engineer / ML Inference Engineer)는 **이미 학습된 AI 모델을 빠르고 싸게 동작시키는 사람**이다.

| 구분 | 하는 일 |
|---|---|
| **ML 리서처** | 새 모델 아키텍처를 만든다 |
| **ML 엔지니어** | 모델을 학습시킨다 (training) |
| **AI 추론 엔지니어** | 학습된 모델을 **서비스로 돌린다 (inference)** |
| **MLOps 엔지니어** | 학습/배포 파이프라인을 자동화한다 |

이 챌린지(`task-ai.md`)는 정확히 **AI 추론 엔지니어**의 역할이다.

## 1.2 시장 현황 (2026년 4월 기준)

### 1.2.1 수요 측면

OpenAI(ChatGPT), Anthropic(Claude), Google(Gemini), xAI 같은 모델 제공자뿐 아니라:

- **모든 SaaS 회사**가 AI 기능을 추가하고 있다 (Notion AI, Slack AI, Zoom AI 등).
- **이미지/영상 생성 SaaS** (Midjourney, Runway, Pika 등)는 AI 추론이 곧 비용의 핵심.
- **스타트업** 수천 개가 자체 추론 인프라를 운영한다.

수요가 폭발하는 이유:
1. 같은 모델을 **싸게 돌리는 회사가 마진을 가져간다**.
2. 같은 모델을 **빠르게 돌리는 회사가 사용자 경험에서 이긴다**.
3. 두 가지 모두 **AI 추론 엔지니어가 만든다**.

### 1.2.2 공급 측면

- **ML 리서처/엔지니어**(모델을 만드는 사람)는 박사·석사 출신이 많아 진입장벽이 높다.
- 그러나 **AI 추론 엔지니어**는 **시스템·인프라 사고**가 더 핵심이다.
- 즉, **백엔드 엔지니어가 진입할 자연스러운 길이다**.

### 1.2.3 연봉 (한국 시장 기준 추정)

> **주의:** 다음 수치는 2026년 4월 기준의 추정 범위이며, 실제 연봉은 회사·연차·협상에 따라 달라진다.

| 연차/위치 | 추정 연봉 (백엔드) | 추정 연봉 (AI 추론) |
|---|---|---|
| 주니어 (1~3년) | 4,500~6,500만 원 | 5,500~8,000만 원 |
| 미들 (4~7년) | 7,000~1억 원 | 9,000~1.5억 원 |
| 시니어 (8년+) | 1억~1.7억 원 | 1.5억~3억 원 + 스톡옵션 |

**AI 추론 엔지니어가 더 높은 이유**:
- 공급이 부족하다.
- 회사가 **모델 추론 비용을 직접 줄이는 사람**이라 ROI가 측정 가능하다.
- 글로벌 회사(NVIDIA, OpenAI, Anthropic) 직접 채용 시 USD 기준 연봉.

## 1.3 직무 정의 — 실제로 무슨 일을 하는가

### 1.3.1 하루 일과 (가상의 예시)

```
09:00  어제 배포한 FlashAttention-3 프로덕션 모니터링
       p95 latency가 850ms → 720ms 로 내려갔는지 확인
       
10:00  오늘 실험 시작
       "DeepCache를 4 step distilled 모델에 적용하면 효과 있을까?"
       Branch 만들고 코드 수정
       
12:00  점심
       
13:00  벤치마크 결과 분석
       속도 1.3배 가속, LPIPS 0.02 증가 (품질 살짝 저하)
       Acceptable한지 PM과 논의
       
15:00  코드 리뷰
       동료가 만든 Triton 커널 PR 리뷰
       
17:00  논문 한 편 정독
       "Sana: Efficient High-Resolution Image Synthesis"
       팀 슬랙에 요약 공유
       
18:00  내일 실험 계획 수립
       퇴근
```

### 1.3.2 사용하는 기술 스택

| 영역 | 도구 |
|---|---|
| 언어 | Python 90%, C++ 10%, CUDA 5% |
| 프레임워크 | PyTorch (압도적 1위), JAX, TensorFlow |
| 최적화 | TensorRT, vLLM, TGI, ONNX Runtime |
| 커널 | Triton, CUTLASS, FlashAttention |
| 정밀도 | TransformerEngine (FP8), bitsandbytes (INT8/4) |
| 측정 | NSight Systems/Compute, torch.profiler, wandb |
| 인프라 | Kubernetes, Docker, GitHub Actions |
| 클라우드 | AWS/GCP/Azure GPU instances, Lambda Labs, RunPod |

### 1.3.3 면접에서 자주 나오는 주제

1. **"이 모델의 추론을 4배 빠르게 하라" — 이 챌린지가 정확히 이 유형이다.**
2. "왜 KV cache가 LLM에서는 되고 디퓨전에서는 안 되는가?"
3. "FP8과 INT8의 차이를 설명하라."
4. "FlashAttention이 빠른 이유를 메모리 계층 관점에서 설명하라."
5. "torch.compile을 사용해본 적 있는가? 어떤 문제를 만났는가?"

이 모든 질문의 답이 이 커리큘럼 안에 있다.

## 1.4 핵심 원리 — 왜 백엔드 출신이 유리한가

### 1.4.1 본질이 같다

AI 추론 엔지니어가 매일 푸는 문제의 본질:

> **"제한된 자원(GPU 메모리, 시간) 안에서 최대 처리량을 뽑아낸다."**

이게 익숙한가? 백엔드가 매일 푸는 문제다.

| 백엔드의 일상 | AI 추론의 일상 |
|---|---|
| DB 쿼리 N+1 문제 | 메모리 IO 병목 |
| Redis 캐시 적중률 | KV cache hit rate |
| Connection pool 튜닝 | Batch size 튜닝 |
| 동시성 제어 | GPU 스레드 모델 |
| p95 latency 최적화 | 동일 |
| Auto-scaling | GPU 인스턴스 스케일링 |

**같은 문제를 다른 도메인에서 풀고 있을 뿐이다.**

### 1.4.2 백엔드가 약한 부분

정직하게 말한다. 백엔드 출신이 처음 부딪히는 벽:

1. **수학** — 행렬, 미적분의 직관 (4장에서 다룬다).
2. **GPU 모델** — CPU 멀티스레딩과 다른 새 패러다임 (5권에서 다룬다).
3. **수치 정밀도** — FP16/FP8 같은 새 어휘 (5권/6권).
4. **모델 내부** — Transformer/디퓨전의 동작 (2권/3권).

이 책의 11권 커리큘럼이 정확히 이 벽들을 하나씩 깨뜨린다.

## 1.5 자가 점검 (Self-Check)

다음 5개 질문에 입으로 답할 수 있으면 1장을 통과한 것이다.

1. AI 추론 엔지니어가 ML 리서처/ML 엔지니어와 무엇이 다른지 한 문장으로 말하라.
2. 시장에서 AI 추론 엔지니어 수요가 폭증하는 이유 두 가지를 들어라.
3. 이 직무의 하루 일과 중 한 가지 시나리오를 자기 말로 묘사하라.
4. AI 추론과 백엔드의 본질이 같은 부분 두 가지를 들어라.
5. 백엔드 출신이 처음 부딪히는 벽 4가지 중 두 가지를 들어라.

---

# 2장. 백엔드 vs AI 추론 — 비슷한 점, 다른 점

## 🎯 이 장의 목적

백엔드와 AI 추론의 **공통점에서 자신감을 얻고**, **차이점을 정확히 짚어** 학습 우선순위를 정한다.

## 2.1 개요

백엔드 엔지니어가 AI 추론으로 옮길 때 가장 큰 심리적 장벽은 **"완전히 다른 세계 같다"**는 느낌이다. 이 장은 이 느낌을 해체한다.

**결론을 먼저 말한다:**

> **본질의 70%는 같고, 표면의 30%만 다르다.**

## 2.2 비슷한 점 (먼저 자신감을 얻자)

### 2.2.1 비교표 — 본질이 같은 항목들

| 비교 항목 | 백엔드 | AI 추론 | 본질 |
|---|---|---|---|
| **요청을 받음** | HTTP request | model.forward(input) | 입력을 받아 출력을 만든다 |
| **응답을 돌려줌** | HTTP response | output tensor | 〃 |
| **느린 부분 찾기** | 슬로우 쿼리 로그 | torch.profiler | 병목 분석 |
| **반복 작업 줄이기** | Redis 캐싱 | KV cache, feature cache | 캐싱 |
| **메모리 절약** | 페이지네이션, 스트리밍 | quantization, chunked inference | 메모리 효율 |
| **동시 처리** | 스레드/async | CUDA stream, batch | 병렬 처리 |
| **자원 부족 시** | 인스턴스 추가 | GPU 추가 | 수평 확장 |
| **느린 종속성 격리** | Circuit breaker | Timeout / fallback | 안정성 |
| **버전 업그레이드** | Blue-green deploy | Canary model deploy | 무중단 배포 |

### 2.2.2 동일한 사고방식

백엔드에서 슬로우 쿼리를 잡을 때:

```
1. EXPLAIN으로 실행 계획 확인
2. 어디서 시간이 가는지 측정
3. 인덱스 추가, 쿼리 재작성, 캐시 도입
4. 다시 측정, regression이 없는지 확인
```

AI 추론에서 모델을 가속할 때:

```
1. torch.profiler로 한 forward의 시간 분포 확인
2. 어디서 시간이 가는지 측정
3. FlashAttention 적용, FP8 양자화, 캐싱 도입
4. 다시 측정, 품질 regression이 없는지 확인
```

**완전히 같은 사고 흐름**이다. 도구만 다르다.

## 2.3 다른 점 (정확히 짚자)

### 2.3.1 차이의 본질 5가지

#### 차이 1 — 자원의 단위

| 백엔드 | AI 추론 |
|---|---|
| CPU core, RAM (GB) | **GPU SM, VRAM (GB)** |
| 1명령 = 1 cycle | 1명령 = 32 thread (warp) |
| 메모리 대역폭 신경 안 씀 | **메모리 대역폭이 핵심** |

GPU는 CPU와 **다른 머신**으로 보아야 한다. 5권 GPU 시스템이 이 차이를 깊게 다룬다.

#### 차이 2 — 입력 데이터의 형태

| 백엔드 | AI 추론 |
|---|---|
| JSON, SQL row, 문자열 | **텐서 (다차원 배열)** |
| 가변 크기 (배열 길이 불특정) | **고정 shape 선호** |
| 스트리밍 가능 | **일괄 처리(batch) 위주** |

shape이 안 맞으면 동작 자체가 안 된다. 이게 처음에 가장 자주 막히는 부분이다.

#### 차이 3 — "동작"의 의미

백엔드:
- 코드가 있으면 정확히 그대로 실행된다.
- 같은 입력 → 같은 출력 (deterministic).

AI 추론:
- 코드가 있어도 **수치 오차**가 있다.
- 같은 입력 → 같은 출력이지만, FP16과 FP32에서 결과가 미세하게 다르다.
- "동작한다"는 **품질 메트릭으로 측정**해야 한다.

#### 차이 4 — 디버깅 방식

| 백엔드 | AI 추론 |
|---|---|
| 로그 + breakpoint | **shape print + 중간 텐서 시각화** |
| 스택 트레이스 | 텐서의 NaN/Inf 검사 |
| Exception | 출력이 "쓰레기 같음" |

NullPointerException 같은 명확한 에러가 적다. 출력 텐서가 "이상함"을 직관으로 알아채야 한다.

#### 차이 5 — 배포의 단위

| 백엔드 | AI 추론 |
|---|---|
| Docker image (~수백 MB) | **모델 파일 (수~수십 GB)** |
| 빠른 cold start (~1초) | **느린 cold start (~수십 초)** |
| 인스턴스 추가 빠름 | **GPU 인스턴스 추가 느림** |

이게 시스템 설계에 큰 영향을 준다 (8권 시스템 설계).

## 2.4 핵심 원리 — 무엇이 정말 새로 배워야 하는가

70% 이미 알고 있다. 새로 배워야 하는 30%는 다음 4개로 압축된다.

```
┌─────────────────────────────────────────────────────┐
│ 백엔드 출신이 새로 배워야 하는 4가지                  │
├─────────────────────────────────────────────────────┤
│ 1. 수학적 직관    — 행렬·미적분의 "감"             │
│ 2. GPU 모델       — CPU와 다른 동시성 패러다임      │
│ 3. 수치 정밀도    — FP16/FP8과 quality trade-off   │
│ 4. 모델 내부      — Transformer/디퓨전의 흐름       │
└─────────────────────────────────────────────────────┘
                       ↓
        이 커리큘럼 11권이 이 4개를 다룬다
```

| 새로 배울 것 | 다루는 권 |
|---|---|
| 수학 | 1권 4장, 5장 |
| GPU | 5권 (전체) |
| 정밀도 | 5권, 6권 |
| 모델 내부 | 2권, 3권, 4권 |

## 2.5 시각자료 — 사고의 다리 그림

```
백엔드 사고                          AI 추론 사고
─────────                          ─────────

[Request 받음]                       [Input tensor 받음]
     │                                    │
     ▼                                    ▼
[Auth, Rate limit]                   [Preprocess (resize, normalize)]
     │                                    │
     ▼                                    ▼
[Service Layer]                      [Model.forward()]
     │     ↓                              │     ↓
     │   [Cache?]                         │   [KV cache?]
     │     │                              │     │
     │     ▼                              │     ▼
     │   [DB query]                       │   [Attention/MLP layers]
     │     │                              │     │
     ▼     ▼                              ▼     ▼
[Response 조립]                       [Output tensor]
     │                                    │
     ▼                                    ▼
[JSON serialize]                     [Postprocess (decode, denormalize)]
     │                                    │
     ▼                                    ▼
[Response 반환]                       [Image/text 반환]

  ▲                                       ▲
  └────── 본질이 같다 ─────────────────────┘
```

## 2.6 자가 점검

1. 백엔드와 AI 추론이 **본질이 같은 항목** 5가지를 들어라.
2. AI 추론의 **자원 단위**가 백엔드와 어떻게 다른가?
3. "동작한다"의 의미가 두 영역에서 어떻게 다른가?
4. 백엔드 출신이 새로 배워야 할 4가지를 들어라.
5. AI 추론 모델의 cold start가 왜 백엔드보다 느린가?

---

# 3장. 어휘 변환표 — 백엔드 ↔ AI/ML

## 🎯 이 장의 목적

AI/ML 영역에서 자주 쓰는 **50개 핵심 용어**를 백엔드 어휘로 매핑하여, 논문·블로그·문서를 읽을 때의 진입 장벽을 해체한다.

## 3.1 개요

처음 ML 논문을 펼치면 외국어처럼 느껴지는 것은 **어휘 자체가 낯설기 때문**이지, 개념이 어렵기 때문이 아니다.

이 장은 **사전(dictionary)** 의 역할이다. 모르는 용어가 나오면 이 장을 펴라.

## 3.2 카테고리 1 — 데이터와 입출력

| AI/ML 용어 | 백엔드 친화 설명 | 예시 |
|---|---|---|
| **Tensor** | 다차원 배열 (NumPy ndarray와 동일) | `shape=(batch, height, width, channel)` |
| **Batch** | 한 번에 처리하는 입력 묶음 | DB의 `WHERE id IN (1,2,3,...)` 같은 일괄 처리 |
| **Sample** | Batch 안의 하나 | 한 사용자 요청 |
| **Feature** | 입력의 한 차원 (변수 하나) | DB row의 한 컬럼 |
| **Label / Target** | 정답 (학습용) | DB의 정답 컬럼 |
| **Embedding** | 정수/문자열을 벡터로 변환한 결과 | hash(word) → 384-dim vector |
| **Tokenization** | 문장을 정수 시퀀스로 자르기 | 문자열 split + ID 매핑 |
| **Padding** | 길이 맞추기 위한 0 채움 | NULL을 0으로 채우는 것 |
| **Mask** | 어떤 위치를 무시할지 표시 | `WHERE deleted_at IS NULL` 같은 필터 |
| **Vocabulary** | 토큰 ID 사전 | enum table |

## 3.3 카테고리 2 — 모델 구조

| AI/ML 용어 | 백엔드 친화 설명 |
|---|---|
| **Layer** | 한 단계의 변환 함수 | middleware 1개 |
| **Neural Network** | Layer들의 chain | middleware chain (Express의 .use().use().use()) |
| **Weight / Parameter** | 학습되는 숫자 | 설정값인데 자동으로 튜닝되는 것 |
| **Bias** | 각 layer의 offset | 함수의 상수항 |
| **Activation** | 비선형 함수 (ReLU 등) | if(x>0) x else 0 |
| **Forward pass** | 입력 → 출력 한 번 | request → response 한 번 |
| **Backward pass** | 학습 중 gradient 계산 | (추론에는 없음) |
| **Inference** | 학습된 모델로 출력 만들기 | "API 호출" |
| **Training** | 모델을 가르치는 과정 | DB 채우기 + 인덱스 빌드 |
| **Checkpoint** | 학습 중 저장된 모델 상태 | DB snapshot |
| **Pretrained model** | 누가 이미 학습해둔 모델 | 외부 라이브러리 |
| **Fine-tuning** | Pretrained 모델을 약간 더 학습 | fork해서 약간 수정 |

## 3.4 카테고리 3 — 추론 최적화

| AI/ML 용어 | 백엔드 친화 설명 |
|---|---|
| **Latency** | 한 요청 처리 시간 | 동일 |
| **Throughput** | 단위 시간당 처리 수 | 동일 (TPS, RPS) |
| **Batch size** | 한 번에 처리하는 sample 수 | bulk insert 사이즈 |
| **KV Cache** | LLM의 attention 결과 저장 | Redis cache |
| **Quantization** | 숫자 정밀도 낮추기 | int 압축 (gzip 비슷) |
| **Pruning** | 모델 일부 제거 | unused index 삭제 |
| **Distillation** | 큰 모델 → 작은 모델로 압축 | 데이터 ETL + 요약 |
| **Compilation** | 모델 → 최적화된 그래프 | JIT compile (V8, LLVM) |
| **Operator Fusion** | 여러 연산 합치기 | query rewriting |
| **Mixed Precision** | FP32와 FP16 섞어 쓰기 | 컬럼별 다른 타입 |

## 3.5 카테고리 4 — GPU

| AI/ML 용어 | 백엔드 친화 설명 |
|---|---|
| **CUDA** | NVIDIA GPU의 표준 API | OS의 syscall 같은 것 |
| **Kernel** | GPU에서 도는 함수 | 작은 람다 함수 |
| **Thread** | GPU 실행 단위 | OS thread보다 훨씬 가벼움 |
| **Warp** | 32 thread 묶음 | SIMD 그룹 |
| **Block** | thread 그룹 | thread pool 한 단위 |
| **Grid** | block 그룹 | 전체 작업 |
| **HBM** | GPU 본 메모리 | RAM (느린 쪽) |
| **SRAM / Shared Memory** | GPU 빠른 캐시 | L1 cache |
| **Tensor Core** | 행렬 곱 전용 유닛 | GPU 안의 ASIC |
| **Stream** | 비동기 명령 큐 | 메시지 큐 |

## 3.6 카테고리 5 — 모델 종류

| AI/ML 용어 | 한 줄 설명 |
|---|---|
| **CNN** | Convolutional Neural Network. 주로 이미지. |
| **RNN / LSTM** | 순차 데이터용 (구식, 거의 안 씀) |
| **Transformer** | 현대 거의 모든 모델의 기반 |
| **BERT** | Encoder-only Transformer (분류용) |
| **GPT** | Decoder-only Transformer (생성용) |
| **ViT** | Vision Transformer (이미지) |
| **DiT** | Diffusion Transformer |
| **VAE** | Variational AutoEncoder (압축) |
| **GAN** | Generative Adversarial Network (구식 생성 모델) |
| **Diffusion** | 노이즈에서 점진적으로 그림 생성 |
| **LLM** | Large Language Model (GPT 같은 거대 모델) |

## 3.7 카테고리 6 — 학습/평가

| AI/ML 용어 | 백엔드 친화 설명 |
|---|---|
| **Loss** | 모델의 틀린 정도 | 에러 메트릭 |
| **Gradient** | Loss를 줄이려면 weight를 어디로? | 미분값 |
| **Optimizer** | Gradient로 weight 업데이트 | 알고리즘 (Adam, SGD) |
| **Learning rate** | 한 번에 얼마나 업데이트할지 | step size |
| **Epoch** | 전체 데이터셋 1회 통과 | 한 cycle |
| **Step / Iteration** | Batch 1개 처리 | 1 트랜잭션 |
| **Overfitting** | 학습 데이터만 잘하고 새 데이터 못함 | 캐시가 너무 stale |
| **Validation** | 학습 중간 점검 | staging test |
| **Test set** | 최종 평가용 데이터 | production smoke test |

## 3.8 자주 보이는 약어 정리

| 약어 | 풀어쓴 용어 | 무엇 |
|---|---|---|
| **DL** | Deep Learning | 신경망 학습 |
| **NN** | Neural Network | 신경망 |
| **MLP** | Multi-Layer Perceptron | 가장 단순한 신경망 |
| **CNN** | Convolutional Neural Network | 이미지용 |
| **FFN** | Feed-Forward Network | Transformer 안의 MLP |
| **MoE** | Mixture of Experts | 여러 모델 중 골라쓰기 |
| **PEFT** | Parameter-Efficient Fine-Tuning | LoRA 같은 가벼운 fine-tuning |
| **RAG** | Retrieval-Augmented Generation | 검색 + 생성 |
| **CFG** | Classifier-Free Guidance | 디퓨전의 가이드 트릭 |
| **EMA** | Exponential Moving Average | 가중 평균 |
| **OOM** | Out Of Memory | 메모리 부족 |
| **OOV** | Out Of Vocabulary | 사전에 없는 단어 |
| **SDP** | Scaled Dot Product (attention) | 표준 어텐션 |
| **MHA** | Multi-Head Attention | 멀티 헤드 어텐션 |
| **GQA** | Grouped Query Attention | 효율적 어텐션 |
| **MLA** | Multi-Layer Attention (DeepSeek) | DeepSeek의 어텐션 |
| **RoPE** | Rotary Position Embedding | 회전 기반 위치 인코딩 |
| **GEMM** | General Matrix Multiply | 행렬 곱 표준 |
| **GEMV** | General Matrix-Vector multiply | 행렬-벡터 곱 |

## 3.9 자가 점검

1. **Tensor**, **Batch**, **Embedding**을 백엔드 용어로 설명하라.
2. **Forward pass**와 **Backward pass**의 차이는?
3. **Inference**와 **Training**의 차이를 한 문장으로.
4. **HBM**과 **SRAM**의 차이를 RAM/L1 cache 비유로 설명하라.
5. **CFG**, **GEMM**, **MHA**가 각각 무엇의 약어인지 답하라.

---

# 4장. 행렬·텐서 완전 정복

## 🎯 이 장의 목적

AI 추론 코드를 한 줄도 막히지 않고 읽을 수 있는 **수학적 어휘**를 갖춘다. 행렬·텐서·shape·broadcasting을 코드 레벨로 익힌다.

> 수학을 두려워할 필요는 없다. **이 장에서 다루는 모든 수학은 백엔드의 N차원 배열로 환원 가능하다.**

## 4.1 개요

신경망 코드의 90%는 **N차원 배열의 변환**이다. 그 중에서도 **행렬 곱**이 핵심이다.

이 장이 끝나면 `(B, H, W, C)` 같은 shape을 보고 자연스럽게 "Batch × Height × Width × Channel" 이라고 읽을 수 있다.

## 4.2 기초 — 0차원부터 N차원까지

### 4.2.1 차원의 진화

```
Scalar (0-dim):       42                              ← 그냥 숫자

Vector (1-dim):       [1, 2, 3]                       ← 배열

Matrix (2-dim):       [[1, 2, 3],                     ← 2D 배열 (행렬)
                       [4, 5, 6]]                       (DB의 row × column)

Tensor (3-dim):       [[[1, 2], [3, 4]],              ← 3D 배열
                       [[5, 6], [7, 8]]]                (이미지 R/G/B 분리)

Tensor (4-dim):       많은 이미지의 배치              ← (B, H, W, C)
                                                        Batch × Height × Width × Channel
```

### 4.2.2 백엔드 비유

| 차원 | 백엔드 비유 |
|---|---|
| Scalar | int 변수 하나 |
| Vector | 1D 배열 (길이 N) |
| Matrix | DB 테이블 (M rows × N cols) |
| 3D Tensor | 여러 테이블의 묶음 |
| 4D Tensor | 여러 테이블의 시계열 |

### 4.2.3 Shape — 가장 중요한 개념

**Shape**: 텐서의 각 차원의 크기를 튜플로 표현한 것.

```python
import numpy as np

a = np.array([1, 2, 3])
print(a.shape)  # (3,)

b = np.array([[1, 2, 3], [4, 5, 6]])
print(b.shape)  # (2, 3)

c = np.zeros((4, 3, 2))
print(c.shape)  # (4, 3, 2)
```

**규칙**: Shape의 길이 = 차원의 수.
**규칙**: Shape의 각 숫자 = 그 차원의 크기.

## 4.3 핵심 원리 — 행렬 곱

### 4.3.1 행렬 곱이 왜 핵심인가

신경망의 한 layer는 본질적으로 **행렬 곱 + bias + 활성화**이다:

```
output = activation(input @ weight + bias)
```

이게 신경망의 99%다. 이걸 정확히 이해하면 거의 모든 코드를 읽을 수 있다.

### 4.3.2 행렬 곱의 규칙

```
A @ B = C

A.shape = (M, K)
B.shape = (K, N)
C.shape = (M, N)

  ↑ K가 같아야 한다 (안 같으면 에러)
```

**시각화**:

```
    A (3 × 4)         B (4 × 2)        C (3 × 2)
   ┌───────┐         ┌─────┐          ┌─────┐
   │ . . . . │   @    │ . . │    =    │ . . │
   │ . . . . │        │ . . │          │ . . │
   │ . . . . │        │ . . │          │ . . │
   └───────┘         │ . . │          └─────┘
                     └─────┘
   M=3, K=4         K=4, N=2          M=3, N=2
```

### 4.3.3 코드 예제

```python
import numpy as np

# 입력 한 sample의 feature vector (길이 4)
x = np.array([1.0, 2.0, 3.0, 4.0])  # shape (4,)

# Weight matrix: 4 input → 2 output
W = np.array([[0.1, 0.2],
              [0.3, 0.4],
              [0.5, 0.6],
              [0.7, 0.8]])  # shape (4, 2)

# Bias
b = np.array([0.01, 0.02])  # shape (2,)

# 한 layer의 forward
output = x @ W + b
print(output)        # [5.01, 6.02]
print(output.shape)  # (2,)
```

이게 신경망 한 layer의 정체다. 끝.

### 4.3.4 Batch 처리

실전에서는 한 번에 여러 sample을 처리한다 (batch).

```python
# Batch 3개의 입력
X = np.array([[1.0, 2.0, 3.0, 4.0],
              [5.0, 6.0, 7.0, 8.0],
              [9.0, 1.0, 2.0, 3.0]])  # shape (3, 4): 3 samples × 4 features

# 같은 W, b 재사용
output = X @ W + b
print(output.shape)  # (3, 2): 3 samples × 2 outputs
```

**핵심**: 첫 번째 차원은 보통 **batch**이고, **그 외 차원만 layer가 처리한다**.

## 4.4 Broadcasting — Shape 자동 맞추기

### 4.4.1 무엇인가

서로 다른 shape의 텐서끼리 연산할 때, NumPy/PyTorch가 **자동으로 작은 쪽을 큰 쪽에 맞춰 확장**해주는 규칙.

### 4.4.2 예제

```python
import numpy as np

# Shape (3, 4)
A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 0, 1, 2]])

# Shape (4,)
b = np.array([10, 20, 30, 40])

# Broadcasting: b가 (1, 4)로 확장되어 모든 row에 더해짐
C = A + b
print(C)
# [[11 22 33 44]
#  [15 26 37 48]
#  [19 20 31 42]]
```

### 4.4.3 Broadcasting 규칙

오른쪽부터 차원을 맞춰가며:
1. 차원이 같으면 OK.
2. 둘 중 하나가 1이면 그쪽이 확장됨.
3. 둘 다 다르면 에러.

```
A.shape = (3, 4)
b.shape =    (4,)         ← 왼쪽에 1이 추가된 것처럼: (1, 4)

A.shape = (3, 4)
b.shape = (3, 1)          ← 두 번째 차원이 1이므로 (3, 4)로 확장
```

### 4.4.4 가장 자주 쓰는 패턴

```python
# Bias를 모든 batch에 더함
output = (X @ W) + b
# X @ W shape: (Batch, Out)
# b shape:     (Out,) → broadcast → 모든 batch에 더함
```

## 4.5 PyTorch에서의 행렬 연산

### 4.5.1 NumPy와 거의 같다

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0, 4.0])
W = torch.tensor([[0.1, 0.2],
                  [0.3, 0.4],
                  [0.5, 0.6],
                  [0.7, 0.8]])
b = torch.tensor([0.01, 0.02])

output = x @ W + b
print(output)  # tensor([5.0100, 6.0200])
```

### 4.5.2 GPU로 옮기기

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_gpu = x.to(device)
W_gpu = W.to(device)
b_gpu = b.to(device)

output = x_gpu @ W_gpu + b_gpu
print(output.device)  # cuda:0 (GPU가 있다면)
```

**주의**: 같은 연산에 들어가는 모든 텐서는 **같은 device** 에 있어야 한다.

## 4.6 자주 나오는 shape 패턴

### 4.6.1 이미지

| 차원 | 의미 | 예시 |
|---|---|---|
| `(H, W, C)` | Height × Width × Channels | OpenCV 표준 |
| `(C, H, W)` | Channels × Height × Width | PyTorch 표준 (channels-first) |
| `(B, C, H, W)` | Batch + 위 | PyTorch 학습/추론 |
| `(B, H, W, C)` | Batch + OpenCV | TensorFlow 표준 |

> **주의**: PyTorch는 channels-first, TensorFlow는 channels-last. 변환 시 주의.

### 4.6.2 시퀀스 (텍스트, 시계열)

| 차원 | 의미 |
|---|---|
| `(B, L)` | Batch × Length (token IDs) |
| `(B, L, D)` | Batch × Length × Dimension (embedding) |
| `(L, B, D)` | seq-first 방식 (구버전 PyTorch) |

### 4.6.3 Attention 계산 안

| 차원 | 의미 |
|---|---|
| `(B, H, L, D)` | Batch × Heads × Length × head_Dim |
| `(B, H, L, L)` | Attention score matrix |

## 4.7 자주 막히는 함정

### 함정 1 — Shape mismatch

```python
A = torch.randn(3, 4)
B = torch.randn(5, 2)
A @ B  # RuntimeError: shape mismatch
```

해결: 항상 `print(x.shape)` 로 확인.

### 함정 2 — Channels-first vs Channels-last

OpenCV로 이미지 읽으면 `(H, W, 3)`이지만, PyTorch 모델은 `(3, H, W)` 기대.

```python
import cv2
img = cv2.imread("photo.jpg")        # shape (H, W, 3) 
img_torch = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)
```

### 함정 3 — Device mismatch

```python
x = torch.tensor([1.0, 2.0]).cuda()
W = torch.tensor([[0.1], [0.2]])  # CPU 위
y = x @ W  # RuntimeError: Expected all tensors to be on the same device
```

해결: 둘 다 `.to(device)`.

### 함정 4 — Dtype mismatch

```python
x = torch.tensor([1.0, 2.0])  # float32
W = torch.tensor([[1, 2], [3, 4]])  # int64
x @ W  # 에러 또는 자동 변환 (성능 저하)
```

해결: 명시적으로 `.float()` 또는 `dtype=torch.float32`.

## 4.8 시각자료 — 행렬 곱의 흐름

```
입력 batch (3 samples × 4 features)
┌──────────┐
│ x x x x │ sample 1
│ x x x x │ sample 2
│ x x x x │ sample 3
└──────────┘
shape (3, 4)
       │
       │ @ W
       ▼
Weight (4 features × 2 outputs)
┌─────┐
│ w w │
│ w w │
│ w w │
│ w w │
└─────┘
shape (4, 2)
       │
       │ + bias
       ▼
Bias (2 outputs,)
[b b]
       │
       ▼
Output (3 samples × 2 outputs)
┌─────┐
│ o o │ sample 1's output
│ o o │ sample 2's output
│ o o │ sample 3's output
└─────┘
shape (3, 2)
```

## 4.9 실습 — 손으로 Layer 한 개 짜보기

다음 코드를 그대로 복붙해서 실행하라.

```python
# practice_4_layer.py
import numpy as np

# 가짜 입력: 5개 sample, 각 sample은 10개 feature
X = np.random.randn(5, 10).astype(np.float32)
print(f"Input shape: {X.shape}")

# Layer 1: 10 input → 8 output
W1 = np.random.randn(10, 8).astype(np.float32) * 0.1
b1 = np.zeros(8, dtype=np.float32)

# Layer 2: 8 input → 4 output
W2 = np.random.randn(8, 4).astype(np.float32) * 0.1
b2 = np.zeros(4, dtype=np.float32)

# Forward
def relu(x):
    return np.maximum(0, x)

h = relu(X @ W1 + b1)
print(f"After Layer 1: {h.shape}")

out = h @ W2 + b2
print(f"Output: {out.shape}")
print(f"Output values:\n{out}")
```

**실행 결과**:
```
Input shape: (5, 10)
After Layer 1: (5, 8)
Output: (5, 4)
Output values:
[[...]]
```

**축하한다.** 방금 신경망 한 forward pass를 손으로 구현한 것이다.

## 4.10 자가 점검

1. `(2, 3, 4)` shape의 텐서는 몇 차원이며, 각 차원의 크기는?
2. `(M, K) @ (K, N)` 의 결과 shape은?
3. `(3, 4)` 와 `(4,)` 를 더하면 결과 shape은? 왜 가능한가?
4. PyTorch 이미지 텐서의 표준 shape 순서는?
5. 위 4.9 실습 코드의 첫 layer를 12 → 16 output으로 바꾸려면 어떻게 수정하는가?

---

# 5장. 신경망의 본질 — "거대한 합성 함수다"

## 🎯 이 장의 목적

신경망의 동작을 **수학적 마법이 아니라 합성 함수**로 본다. "왜 이게 작동하는가"의 직관을 얻는다.

## 5.1 개요

신경망(Neural Network)은 다음 한 줄로 요약 가능하다:

> **여러 단순한 함수를 깊게 쌓아서 복잡한 함수를 근사한다.**

이게 전부다. "뉴런", "시냅스" 같은 생물학적 비유는 잊어라. 결국 **함수**다.

## 5.2 기초 — 함수 합성으로 시작

### 5.2.1 단순 함수에서 출발

```python
def f(x):
    return 2 * x + 1

def g(x):
    return x * x

# 합성: g(f(x))
def h(x):
    return g(f(x))

print(h(3))  # f(3)=7, g(7)=49
```

신경망의 본질: `output = layer_N( layer_{N-1}( ... layer_1(input) ... ) )`.

### 5.2.2 Layer 하나의 정체

```python
def linear_layer(x, W, b):
    return x @ W + b

def relu(x):
    return np.maximum(0, x)

# Layer 1
def layer_1(x):
    return relu(linear_layer(x, W1, b1))

# Layer 2
def layer_2(x):
    return relu(linear_layer(x, W2, b2))

# Network = layer_2(layer_1(x))
```

위에서 `linear_layer + relu` 가 한 layer의 본질이다.

### 5.2.3 백엔드 비유 — Middleware Chain

Express.js를 안다면:

```javascript
app.use(authMiddleware)
   .use(rateLimiter)
   .use(parseJson)
   .use(handler);
```

각 middleware가 request를 받아 변환해서 다음으로 넘긴다.

신경망도 같다:

```python
output = layer_5(layer_4(layer_3(layer_2(layer_1(input)))))
```

각 layer가 텐서를 받아 변환해서 다음으로 넘긴다.

**"middleware = layer"** 로 놓고 보면 직관이 즉시 선다.

## 5.3 핵심 원리 — 비선형성이 핵심이다

### 5.3.1 선형만 쌓으면 의미 없다

만약 모든 layer가 단순 행렬 곱이라면:

```
output = ((x @ W1) @ W2) @ W3
       = x @ (W1 @ W2 @ W3)
       = x @ W_combined
```

→ **여러 layer를 쌓아도 결국 행렬 곱 한 번과 같다.** 의미 없다.

### 5.3.2 활성화 함수의 역할

`ReLU`, `Sigmoid`, `GELU` 같은 **비선형 함수**를 사이에 끼워야 의미 있는 모델이 된다.

```python
output = relu(relu(relu(x @ W1 + b1) @ W2 + b2) @ W3 + b3)
```

이렇게 하면 **임의의 함수를 근사 가능**해진다 (Universal Approximation Theorem).

### 5.3.3 가장 자주 쓰는 활성화 함수

```python
def relu(x):
    return np.maximum(0, x)              # 음수 → 0, 양수 → 그대로

def sigmoid(x):
    return 1 / (1 + np.exp(-x))           # 0~1 사이로 압축

def tanh(x):
    return np.tanh(x)                     # -1~1 사이로 압축

def gelu(x):
    # 현대 Transformer의 표준
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
```

**시각자료**:

```
ReLU:                   GELU:                   Sigmoid:
                         
   y                     y                       y
   │                     │                       │
   │      ╱              │      ╱              1.0─────
   │    ╱                │    ╱  ╱             │      ╱
   │  ╱                  │  ╱    ╱             │   ╱╱
───┼────── x         ────┼─╱─── x          ────┼╱───── x
   │                     │ ╱                   │
   │                    ─┤                    0.0
   │                     │                       │
```

## 5.4 자세한 내용 — 깊이의 의미

### 5.4.1 왜 깊게 쌓는가

**얕은 네트워크**: 함수 1개를 통과 → 단순한 패턴만 학습.

**깊은 네트워크**: 함수 N개 통과 → 복잡한 추상화 가능.

이미지 분류 예시:
```
Layer 1: 픽셀 → 가장자리(edge)
Layer 2: 가장자리 → 모서리, 곡선
Layer 3: 모서리 → 눈, 코, 입의 부분
Layer 4: 부분 → 얼굴 전체
Layer 5: 얼굴 → 사람의 ID
```

각 layer가 **점점 추상적인 특징** 을 추출한다.

### 5.4.2 하지만 깊어지면 학습이 어렵다

문제:
- Gradient가 layer를 거슬러 올라가면서 사라진다 (vanishing gradient).
- 또는 폭발한다 (exploding gradient).

해결책:
- **Residual connection (ResNet, 2015)**: `output = layer(x) + x`. 그래디언트가 우회로로 흐른다.
- **Layer Normalization / Batch Normalization**: 분포를 안정화.

이 두 가지가 **깊은 네트워크를 학습 가능하게 만든** 결정적 발명이다. Transformer 안에도 모두 들어있다.

### 5.4.3 코드 — 진짜 신경망 한 개 만들기

```python
# practice_5_mlp.py
import numpy as np

np.random.seed(42)

# 입력: 28x28 이미지 평탄화 → 784
INPUT_DIM = 784
HIDDEN_DIM = 128
OUTPUT_DIM = 10  # 0~9 분류

# 초기화
W1 = np.random.randn(INPUT_DIM, HIDDEN_DIM).astype(np.float32) * 0.01
b1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
W2 = np.random.randn(HIDDEN_DIM, OUTPUT_DIM).astype(np.float32) * 0.01
b2 = np.zeros(OUTPUT_DIM, dtype=np.float32)

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    """확률 분포로 변환 (마지막 layer)"""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))  # 수치 안정성
    return e / e.sum(axis=-1, keepdims=True)

def forward(x):
    h = relu(x @ W1 + b1)
    logits = h @ W2 + b2
    probs = softmax(logits)
    return probs

# 가짜 batch: 32 images
X = np.random.randn(32, INPUT_DIM).astype(np.float32)
probs = forward(X)
print(f"Output shape: {probs.shape}")           # (32, 10)
print(f"Probs sum per sample: {probs.sum(1)}")  # 각 row 합 = 1
print(f"Predicted class for sample 0: {probs[0].argmax()}")
```

**실행 결과**:
```
Output shape: (32, 10)
Probs sum per sample: [1.0 1.0 1.0 ... 1.0]
Predicted class for sample 0: 7
```

이게 **MNIST 숫자 분류기의 forward pass**다. 학습은 안 된 상태(랜덤 W)지만, 구조는 진짜 모델과 동일하다.

## 5.5 한 걸음 더 — Loss와 학습의 직관 (추론에는 필요 없지만 알아두자)

학습이란?

```
1. forward로 예측
2. 정답과 비교 → loss 계산
3. loss가 줄어들도록 W를 살짝 조정 (gradient descent)
4. 1~3 반복
```

추론 엔지니어는 **이미 학습된 모델을 받아서** forward만 한다. 학습은 ML 엔지니어가 한다.

> **이 챌린지(`task-ai`)는 학습이 아니라 추론 최적화다.** 그래서 backward / gradient는 신경 안 써도 된다.

## 5.6 핵심 정리

```
┌───────────────────────────────────────────────────────┐
│  신경망 = 합성 함수 + 비선형성 + 깊이                   │
│                                                         │
│  output = activation(layer_N(...activation(layer_1(x))))│
│                                                         │
│  → 학습은 W를 찾는 일                                   │
│  → 추론은 W가 정해진 상태에서 forward만 돌리는 일       │
│                                                         │
│  AI 추론 엔지니어 = forward를 빠르게 돌리는 사람         │
└───────────────────────────────────────────────────────┘
```

## 5.7 자가 점검

1. 신경망의 한 줄 정의를 본인 말로 표현하라.
2. 왜 비선형 활성화 함수가 필요한가?
3. 깊은 네트워크의 학습이 어려운 이유 두 가지와 해결책을 들어라.
4. ReLU 함수를 1줄 코드로 작성하라.
5. 5.4.3의 코드에서 hidden dim을 256으로 바꾸려면 어디를 수정하는가?

---

# 6장. PyTorch 30분 입문

## 🎯 이 장의 목적

PyTorch의 **핵심 객체 3개** (Tensor, nn.Module, Optimizer)를 익혀, 모든 ML 코드를 읽을 수 있게 된다. 추론 엔지니어는 Optimizer는 거의 안 쓰지만 한 번은 봐둔다.

## 6.1 개요

PyTorch는 **사실상의 표준 딥러닝 프레임워크**이다 (2026년 기준 시장 점유율 80%+).

| 백엔드 | AI 추론 |
|---|---|
| Spring / Express / Django | **PyTorch** |
| ORM (JPA, SQLAlchemy) | torch.nn |
| HTTP client | torch.jit / torch.compile |

## 6.2 기초 — Tensor

### 6.2.1 NumPy의 GPU 버전이라고 보면 된다

```python
import torch

# 텐서 생성
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.zeros(3, 4)
c = torch.randn(2, 3, 4)  # 정규분포

# Shape
print(a.shape)  # torch.Size([3])
print(c.shape)  # torch.Size([2, 3, 4])

# 연산
d = a + 1
e = a @ b
```

### 6.2.2 Device 이동

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
a_gpu = a.to(device)
```

### 6.2.3 Dtype

```python
a = torch.tensor([1.0, 2.0], dtype=torch.float32)
b = a.half()      # float16
c = a.bfloat16()  # bfloat16
```

추론 엔지니어가 dtype을 자유자재로 다루는 것은 **필수 스킬**이다.

## 6.3 핵심 원리 — nn.Module

### 6.3.1 신경망의 표준 추상화

```python
import torch
import torch.nn as nn

class MyMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MyMLP(784, 128, 10)
print(model)
```

**출력**:
```
MyMLP(
  (fc1): Linear(in_features=784, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=10, bias=True)
  (relu): ReLU()
)
```

### 6.3.2 forward 호출

```python
x = torch.randn(32, 784)  # batch 32
output = model(x)         # 자동으로 forward() 호출
print(output.shape)       # torch.Size([32, 10])
```

`model(x)` 가 `model.forward(x)` 와 같다. 단, 부가 동작(hook 등) 처리 때문에 **항상 `model(x)`를 사용**한다.

### 6.3.3 백엔드 비유 — Service Class

| Spring/Express | PyTorch |
|---|---|
| `@Service` 클래스 | `nn.Module` 클래스 |
| 의존성 주입 (필드) | sub-module (필드) |
| public method | `forward()` |
| `service.execute()` | `model(input)` |

같은 객체 지향 설계다.

## 6.4 자세한 내용 — 자주 쓰는 nn 모듈

### 6.4.1 Linear (완전연결 layer)

```python
nn.Linear(in_features, out_features, bias=True)
# 내부: output = input @ W + b
```

### 6.4.2 Conv2d (이미지용)

```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
# 입력: (B, C_in, H, W)
# 출력: (B, C_out, H', W')
```

### 6.4.3 LayerNorm / BatchNorm

```python
nn.LayerNorm(normalized_shape)
nn.BatchNorm2d(num_features)
```

### 6.4.4 Activation 함수

```python
nn.ReLU(), nn.GELU(), nn.SiLU(), nn.Sigmoid()
```

### 6.4.5 Embedding

```python
nn.Embedding(vocab_size, embedding_dim)
# 정수 ID → 벡터
```

### 6.4.6 Multi-Head Attention

```python
nn.MultiheadAttention(embed_dim, num_heads)
```

이게 Transformer의 핵심이다. 3권에서 깊이 다룬다.

## 6.5 추론 모드 — `.eval()` 과 `torch.no_grad()`

추론할 때 두 가지를 반드시 한다:

```python
model.eval()                   # Dropout/BatchNorm을 추론 모드로
with torch.no_grad():           # Gradient 계산 끔 (메모리 절약)
    output = model(x)
```

**이유**:
- `model.eval()`: dropout이 활성/비활성에 따라 다르게 동작하기 때문.
- `torch.no_grad()`: 학습 안 하니까 gradient 계산은 메모리 낭비.

추론 엔지니어가 가장 먼저 챙기는 두 줄이다.

## 6.6 모델 저장과 불러오기

```python
# 저장
torch.save(model.state_dict(), "my_model.pt")

# 불러오기
model = MyMLP(784, 128, 10)
model.load_state_dict(torch.load("my_model.pt"))
model.eval()
```

`state_dict` = 모든 weight의 딕셔너리. JSON 같은 직렬화 가능 형태.

## 6.7 시각자료 — PyTorch 워크플로우

```
┌──────────────────────────────────────────────────┐
│         PyTorch 추론 워크플로우 (실전)            │
└──────────────────────────────────────────────────┘

[1] 모델 클래스 정의 (nn.Module)
       │
       ▼
[2] 사전학습 weight 불러오기
       │  (model.load_state_dict(torch.load(...)))
       ▼
[3] device로 이동
       │  (model.to("cuda"))
       ▼
[4] eval mode 전환
       │  (model.eval())
       ▼
[5] no_grad 컨텍스트
       │  (with torch.no_grad():)
       ▼
[6] 입력 전처리
       │  (resize, normalize, to(device))
       ▼
[7] forward
       │  (output = model(x))
       ▼
[8] 후처리
       │  (decode, denormalize, .cpu().numpy())
       ▼
[9] 결과 반환
```

이 9단계를 외워두면 모든 추론 코드 패턴이 보인다.

## 6.8 실습 — 6장 전체 결합

```python
# practice_6_pytorch_mlp.py
import torch
import torch.nn as nn

# 1. 모델 정의
class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# 2. 인스턴스화
model = TinyMLP()

# 3. (보통은 weight load) — 여기선 랜덤 그대로 사용
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# 4. 가짜 입력
batch = torch.randn(4, 784).to(device)

# 5. 추론
with torch.no_grad():
    output = model(batch)

print(f"Output shape: {output.shape}")  # torch.Size([4, 10])
print(f"Predicted classes: {output.argmax(dim=-1).tolist()}")
```

## 6.9 자가 점검

1. PyTorch에서 GPU로 텐서를 옮기는 코드 한 줄을 작성하라.
2. `nn.Module`의 핵심 메서드 이름은?
3. 추론 시 반드시 호출해야 하는 모드 전환 코드와, 감싸야 할 컨텍스트는?
4. `state_dict`가 무엇인가?
5. 6.8 실습의 모델에서 hidden dim을 256으로 바꾸고, batch를 8로 바꿔보라.

---

# 7장. 첫 모델 추론 — Hugging Face

## 🎯 이 장의 목적

**한 시간 안에**, 사전학습 모델을 다운로드하여 실제 추론 한 번을 실행한다. 손으로 한 번 돌려본 사람과 안 돌려본 사람의 차이는 크다.

## 7.1 개요

**Hugging Face Hub**: 수십만 개의 사전학습 모델이 모여있는 저장소.

| 백엔드 | AI 추론 |
|---|---|
| npm registry | **Hugging Face Hub** |
| `npm install lodash` | `from transformers import AutoModel` |
| package.json | `huggingface_hub`, `transformers` |

## 7.2 설치

```bash
pip install torch transformers diffusers accelerate
```

GPU가 있다면 CUDA 버전에 맞는 PyTorch를 설치한다 ([부록 A](#부록-a--환경-셋업--python-pytorch-cuda) 참고).

## 7.3 핵심 원리 — `pipeline` API

가장 빠른 길:

```python
from transformers import pipeline

# 감정 분석 모델 자동 다운로드 + 사용
classifier = pipeline("sentiment-analysis")

result = classifier("I love this curriculum!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998...}]
```

이 3줄로 **사전학습 모델 다운로드 + 토큰화 + 추론 + 후처리**가 끝났다.

## 7.4 자세히 — 한 단계씩 들여다보기

`pipeline`이 자동으로 하던 것을 풀어쓴다.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 1. 토크나이저 (문자열 → 정수 ID)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# 2. 모델
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model.eval()

# 3. 입력 토큰화
text = "I love this curriculum!"
inputs = tokenizer(text, return_tensors="pt")
# inputs = {"input_ids": tensor([[101, 1045, 2293, ...]]), "attention_mask": tensor([[1, 1, 1, ...]])}

# 4. 추론
with torch.no_grad():
    outputs = model(**inputs)

# 5. 결과 해석
logits = outputs.logits           # shape (1, 2)
probs = logits.softmax(dim=-1)
predicted_class = probs.argmax(-1).item()
labels = ["NEGATIVE", "POSITIVE"]
print(f"{labels[predicted_class]} ({probs[0, predicted_class]:.4f})")
# POSITIVE (0.9998)
```

**6장의 9단계 워크플로우**를 그대로 따랐다.

## 7.5 이미지 생성 — Stable Diffusion 1.5

이 챌린지의 직접적인 친척이다. 한 번 돌려보자.

### 7.5.1 설치 확인

```bash
pip install diffusers transformers accelerate
```

### 7.5.2 코드

```python
# practice_7_sd.py
import torch
from diffusers import StableDiffusionPipeline

# 1. 모델 로드 (첫 실행 시 약 5GB 다운로드)
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,            # FP16 (메모리 절반)
)

# 2. GPU로 이동
pipe.to("cuda")

# 3. 추론
prompt = "a photograph of an astronaut riding a horse"
image = pipe(prompt, num_inference_steps=20).images[0]

# 4. 저장
image.save("astronaut_horse.png")
print("Saved!")
```

### 7.5.3 결과

`astronaut_horse.png` 가 만들어진다.

**축하한다.** 방금 디퓨전 모델 추론을 돌린 것이다. 이 챌린지의 FLUX.2도 같은 패밀리다.

### 7.5.4 GPU가 없다면

CPU로도 돌아간다. 단, 한 장에 5~10분 걸린다.

```python
pipe.to("cpu")  # 그냥 CPU 사용
# torch_dtype=torch.float16 대신 float32로 변경
```

또는 **Google Colab의 무료 T4 GPU**를 사용하라.

## 7.6 한 걸음 더 — 직접 forward 호출

`pipe(prompt)`가 자동으로 한 일을 풀어쓰면:

```python
# 1. Text encoder: prompt → text embedding
text_input = pipe.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
text_emb = pipe.text_encoder(text_input)[0]

# 2. 노이즈 초기화
import torch
latents = torch.randn(1, 4, 64, 64, dtype=torch.float16).to("cuda")

# 3. Denoising loop (간략화)
pipe.scheduler.set_timesteps(20)
for t in pipe.scheduler.timesteps:
    with torch.no_grad():
        noise_pred = pipe.unet(latents, t, encoder_hidden_states=text_emb).sample
    latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

# 4. VAE decode: latent → image
with torch.no_grad():
    image_tensor = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample

# 5. Tensor → PIL image
from PIL import Image
import numpy as np
image_np = (image_tensor.cpu().permute(0, 2, 3, 1).numpy()[0] * 0.5 + 0.5).clip(0, 1)
image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
image_pil.save("manual.png")
```

이 코드를 이해하면 **이 챌린지의 80%는 이미 본 것**이다.
- text encoder, scheduler, unet, vae — 모두 FLUX.2에도 있다.
- denoising loop — 이게 챌린지의 "denoise ×4".
- VAE decode — 챌린지의 마지막 단계.

## 7.7 시각자료 — Stable Diffusion 추론 흐름

```
"a photograph of an astronaut riding a horse"
            │
            ▼
   ┌─────────────────┐
   │  CLIP Tokenizer │   (text → IDs)
   └─────────────────┘
            │
            ▼
   ┌─────────────────┐
   │  CLIP Text      │   (IDs → embedding)
   │  Encoder         │
   └─────────────────┘
            │
            │  ┌─────────────────┐
            │  │ Random noise    │ (latent)
            │  │ shape (1,4,64,64)│
            │  └─────────────────┘
            ▼          │
   ┌─────────────────┐ │
   │   UNet (×20 step)│←┘
   │   (denoise loop) │
   └─────────────────┘
            │
            ▼ (denoised latent)
   ┌─────────────────┐
   │   VAE Decoder   │   (latent → 512×512 image)
   └─────────────────┘
            │
            ▼
       [최종 이미지]
```

이 그림이 이 챌린지(`task-ai`)의 그림과 거의 동일하다. FLUX.2는:
- UNet → DiT (Transformer 기반)
- 20 step → 4 step (distilled)
- 추가로 reference image 입력

이 변형들이 2~4권에서 다뤄질 내용이다.

## 7.8 자가 점검

1. Hugging Face Hub은 백엔드의 무엇과 비슷한가?
2. `pipeline()` 한 줄에 자동으로 일어나는 일을 5단계로 분해하라.
3. SD 추론 코드의 9단계 워크플로우를 6장의 그림과 매칭시켜라.
4. 7.6의 manual SD 코드에서 `denoising loop`가 몇 번 돌도록 설정되어 있는가? 4번으로 줄이려면?
5. GPU 없이 SD를 돌릴 수 있는가? 어떻게?

---

# 8장. AI 추론 서비스 vs 일반 REST API

## 🎯 이 장의 목적

AI 추론 서비스가 일반 REST API와 무엇이 다른지 **5가지 관점**에서 정리한다. 시스템 설계의 시야가 열린다.

## 8.1 개요

AI 추론 서비스도 결국 HTTP를 받아 응답하는 서버다. 그러나 다음 5가지가 **결정적으로 다르다**.

```
┌──────────────────────────────────────────┐
│  AI 추론 서비스의 5가지 차이              │
├──────────────────────────────────────────┤
│  1. Cold start 비용                      │
│  2. 자원 할당 단위                        │
│  3. Batching의 의미                       │
│  4. 처리 시간의 분포                      │
│  5. 모델 버전 관리                        │
└──────────────────────────────────────────┘
```

## 8.2 차이 1 — Cold Start 비용

### 8.2.1 일반 REST API
- 서버 띄우는 데 1~5초.
- Lambda cold start: ~수백 ms.
- 별 신경 안 써도 된다.

### 8.2.2 AI 추론
- **모델 파일 로드**: 5GB 모델 → 디스크에서 GPU 메모리로 옮기는 데 10~30초.
- **CUDA context 초기화**: 추가 2~5초.
- **첫 forward**: warmup 필요 (커널 컴파일, autotuning) → 5~30초.
- **합계: 30초~1분 cold start**.

### 8.2.3 시스템 설계 함의

| 전략 | 설명 |
|---|---|
| **항상 켜두기 (always-on)** | Lambda 못 씀. 인스턴스 미리 띄워둔다. |
| **Pre-warm pool** | 여분 인스턴스 항상 준비. |
| **Pre-load weight** | 컨테이너 이미지에 weight 미리 포함. |
| **Slow scale** | 트래픽 폭증에 즉시 대응 못함. 미리 스케일. |

## 8.3 차이 2 — 자원 할당 단위

### 8.3.1 일반 REST API
- CPU core, RAM 단위.
- 한 인스턴스에 여러 요청 동시 처리 (thread pool).
- 가벼운 요청은 millisecond 단위.

### 8.3.2 AI 추론
- **GPU 단위**.
- GPU 한 대에 한 모델 1개~몇 개 (메모리 한계).
- **GPU는 한 번에 한 batch만 처리** (시리얼).
- 한 요청이 secs ~ minutes.

### 8.3.3 시각자료

```
일반 REST API 인스턴스:
┌─────────────┐
│  CPU 8 core │
│  RAM 16GB   │
│             │
│  [Req 1]    │  ← 동시
│  [Req 2]    │  ← 동시
│  [Req 3]    │  ← 동시
│  ...        │  ← 동시
└─────────────┘

AI 추론 인스턴스 (GPU):
┌─────────────┐
│  CPU 8 core │
│  RAM 32GB   │
│  GPU 1대     │
│  VRAM 80GB  │
│             │
│  [Req 1] → 처리 중   │
│  [Req 2] ← 큐 대기   │
│  [Req 3] ← 큐 대기   │
└─────────────┘
```

## 8.4 차이 3 — Batching의 의미

### 8.4.1 일반 REST API
- "Batch" = bulk insert 같은 일괄 처리 (선택적).
- 단일 요청 처리에는 보통 안 씀.

### 8.4.2 AI 추론
- **Batching이 핵심 최적화 수단**.
- GPU는 batch size 1보다 32일 때 처리량이 5~20배.
- 그러나 **latency는 batch가 클수록 길어짐**.

### 8.4.3 Continuous Batching

vLLM의 핵심 아이디어:
- 요청을 모아 batch로 묶음.
- 새 요청이 오면 진행 중 batch에 즉석 합류.
- LLM처럼 step별 처리 모델에 적합.

8권 시스템 설계에서 깊게 다룬다.

## 8.5 차이 4 — 처리 시간의 분포

### 8.5.1 일반 REST API
- 처리 시간 분산이 작다 (대부분 비슷).
- p50과 p99 차이 보통 10~100배 이내.

### 8.5.2 AI 추론
- 입력 길이/복잡도에 따라 **수백 배 차이 가능**.
- 특히 LLM의 **생성 토큰 수**가 0~수천으로 가변.
- p50과 p99 차이 1000배도 가능.

### 8.5.3 시스템 설계 함의

- Timeout 설계가 까다롭다.
- Tail latency 최적화가 어렵다.
- Streaming 응답 (SSE, WebSocket) 필수일 때가 많다.

## 8.6 차이 5 — 모델 버전 관리

### 8.6.1 일반 REST API
- 코드 deploy → 새 버전.
- Blue-green / canary 가 표준.

### 8.6.2 AI 추론
- **모델 자체가 큰 binary** (5~수백 GB).
- 새 모델 배포 = 큰 파일 전송 + warmup.
- A/B 테스트 = 두 모델을 동시 띄움 → 메모리 2배.
- **모델 버전 + 코드 버전** 둘 다 추적 필요.

### 8.6.3 도구

- **Hugging Face Hub** + `revision` 인자.
- **MLflow Model Registry**.
- **Weights & Biases (wandb)**.
- 자체 S3 + metadata DB.

## 8.7 비교표 종합

| 측면 | 일반 REST API | AI 추론 서비스 |
|---|---|---|
| 인스턴스 단위 | CPU + RAM | **GPU + VRAM** |
| Cold start | 1~5초 | **30초~1분** |
| 배포 단위 | Docker 이미지 | **모델 + 코드** |
| 단일 요청 시간 | ms | **초~분** |
| 동시 처리 모델 | thread pool | **batch / queue** |
| Latency 분포 | 좁음 | **매우 넓음** |
| Auto-scaling | 빠름 | **느림** |
| 리소스 비용 | 시간당 $0.01~$1 | **시간당 $1~$30** |
| 캐싱 단위 | DB query, CDN | **KV cache, feature cache** |

## 8.8 핵심 원리

```
┌───────────────────────────────────────────────────┐
│  AI 추론 서비스는 "비싼 자원(GPU)을 효율적으로     │
│  공유"하는 시스템 설계가 본질이다.                 │
│                                                     │
│  → Cold start는 피하라 (always-on)                 │
│  → Batch로 처리량을 높여라                         │
│  → Cache로 중복을 제거하라                          │
│  → Tail latency를 모니터링하라                     │
│  → 모델 버전을 코드와 함께 추적하라                 │
└───────────────────────────────────────────────────┘
```

## 8.9 자가 점검

1. AI 추론 서비스의 cold start가 일반 REST API보다 왜 느린가? 두 가지 이유.
2. GPU 인스턴스가 한 번에 한 batch만 처리하는 이유는?
3. Batching이 latency와 throughput에 어떻게 다르게 영향을 주는가?
4. AI 추론의 tail latency가 큰 이유는?
5. AI 모델 배포가 일반 코드 배포보다 까다로운 이유 두 가지를 들어라.

---

# 9장. 모델의 라이프사이클

## 🎯 이 장의 목적

데이터 수집부터 추론·모니터링까지 **모델의 전 생애**를 백엔드의 CI/CD와 비교하여 이해한다. 추론 엔지니어가 어디에 위치하는지 명확히 한다.

## 9.1 개요

```
[데이터] → [학습] → [평가] → [변환] → [배포] → [추론] → [모니터링] → 다시 데이터
   │         │       │        │       │       │           │
ML            ML       ML       ML/추론  추론   추론       MLOps
엔지니어      엔지니어  엔지니어 엔지니어 엔지니어 엔지니어   엔지니어
```

**추론 엔지니어**가 책임지는 부분: 변환, 배포, 추론, 일부 모니터링.

## 9.2 단계별 상세

### 9.2.1 데이터 수집 (Data Collection)

ML 엔지니어 또는 데이터 엔지니어의 일.

- 학습용 데이터 모으기.
- 라벨링 (사람이 정답 다는 작업).
- 정제 (중복 제거, 노이즈 제거).

**추론 엔지니어 관여**: 거의 없음. 단, 추론 시점의 입력 분포가 학습 데이터와 다르면 문제(distribution drift) 가 생기므로 모니터링.

### 9.2.2 학습 (Training)

```python
# 학습 코드의 본질
for epoch in range(num_epochs):
    for batch in dataloader:
        x, y = batch
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()      # gradient 계산
        optimizer.step()     # weight 업데이트
        optimizer.zero_grad()
```

- Forward + backward + optimize 반복.
- 보통 **수일~수개월** 소요.
- GPU 수백~수만 대.

**추론 엔지니어 관여**: 거의 없음. 학습된 weight를 받아온다.

### 9.2.3 평가 (Evaluation)

- 학습된 모델이 새 데이터에서 잘 동작하는가.
- Test set으로 메트릭 측정.

**추론 엔지니어 관여**: 추론 최적화가 품질에 영향을 주지 않는지 회귀 테스트.

### 9.2.4 변환 (Conversion / Export)

여기서부터 **추론 엔지니어의 영역**.

- PyTorch model → ONNX
- ONNX → TensorRT engine
- 또는 PyTorch → torch.compile / torch.jit

```python
# 예: ONNX export
torch.onnx.export(
    model,
    sample_input,
    "model.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)
```

### 9.2.5 배포 (Deployment)

- 모델 파일을 GPU 서버에 올림.
- HTTP/gRPC 서버 띄움.
- 인프라 모니터링 연결.

**도구**:
- **NVIDIA Triton Inference Server**: 표준 모델 서빙.
- **vLLM**: LLM 특화.
- **TGI (Text Generation Inference)**: HuggingFace의 LLM 서버.
- **TorchServe**: PyTorch 공식.
- 자체 FastAPI 래퍼.

### 9.2.6 추론 (Inference)

- 사용자 요청을 받아 모델 forward.
- 후처리.
- 응답.

이 챌린지가 정확히 이 단계의 최적화다.

### 9.2.7 모니터링 (Monitoring)

- Latency, throughput.
- 에러율.
- **모델 품질** (회귀 감지).
- **분포 drift** (입력이 학습 데이터와 달라지는지).

## 9.3 백엔드의 CI/CD와 비교

| 백엔드 | AI 추론 |
|---|---|
| 코드 작성 | 모델 학습 |
| 단위 테스트 | 모델 평가 (validation) |
| 빌드 | 모델 변환 (ONNX/TensorRT) |
| 통합 테스트 | end-to-end 추론 테스트 |
| Staging 배포 | 모델 staging serving |
| Canary | 일부 트래픽만 새 모델로 |
| Production deploy | 모델 production serving |
| 로그/메트릭 | latency, quality 메트릭 |
| 롤백 | 이전 모델 버전으로 |

**구조가 완전히 같다.** 단위가 다를 뿐.

## 9.4 시각자료 — 라이프사이클 그림

```
┌──────────────────────────────────────────────────┐
│              모델 라이프사이클                    │
└──────────────────────────────────────────────────┘

  [데이터] ──→ [학습] ──→ [평가]
                                │
                                ▼
                         [Checkpoint 저장]
                                │
                                ▼  ← 추론 엔지니어 진입
                         [변환 (ONNX/TensorRT)]
                                │
                                ▼
                         [추론 서버 배포]
                                │
                                ▼
   ┌──→ [실시간 추론 (사용자 요청)]
   │            │
   │            ▼
   │    [응답 + 로그]
   │            │
   │            ▼
   │    [모니터링: latency, 품질, drift]
   │            │
   │            ▼
   └──── [회귀 발견 시 롤백 또는 재학습]
```

## 9.5 자가 점검

1. 모델 라이프사이클 7단계를 순서대로 들어라.
2. 추론 엔지니어가 책임지는 단계는 어디부터인가?
3. 모델 변환에서 PyTorch가 거치는 보통의 변환 단계는?
4. AI 모니터링이 일반 백엔드 모니터링보다 추가로 봐야 하는 두 가지는?
5. "Distribution drift"가 무엇이며 왜 위험한가?

---

# 10장. 백엔드 엔지니어가 가진 강점

## 🎯 이 장의 목적

백엔드 출신이 AI 추론으로 갈 때 **이미 가진 무기**를 명시적으로 인식한다. 자신감의 근거를 만든다.

## 10.1 개요

ML 박사 출신은 모델을 잘 만든다. 그러나 **추론 시스템 전체를 책임지는 능력**은 백엔드 출신이 더 강할 때가 많다. 다음 5가지가 결정적 강점이다.

## 10.2 강점 1 — 시스템 설계 사고

ML 출신이 약한 부분:
- 분산 시스템.
- 메시지 큐, 이벤트 드리븐.
- 캐싱 계층 설계.
- 부하 분산.

백엔드 출신은 이 모든 것을 이미 안다.

**예시**:
> AI 추론 서비스의 응답 시간이 느릴 때, ML 출신은 모델 자체를 더 줄이려 한다.
> 백엔드 출신은 **"앞에 결과 캐시 한 줄 두면 되겠는데?"** 를 즉시 떠올린다.

이 챌린지(`task-ai`) 의 핵심 통찰도 정확히 그것이다: **"아바타가 안 변하니 캐시하자"**.

## 10.3 강점 2 — 동시성과 비동기

백엔드는:
- async/await을 자유자재로.
- thread pool, connection pool의 직관.
- 자원 contention 사고.

GPU 인프라:
- CUDA stream = async I/O.
- Batching = 동시 처리.
- Queue depth = pool size.

**같은 사고 패턴**.

## 10.4 강점 3 — 운영 사고 (SRE 마인드)

ML 출신이 자주 놓치는 것:
- p95/p99 latency.
- 장애 시 fallback.
- 점진적 배포 (canary).
- On-call 대응.

백엔드는 이 모든 것을 매일 한다.

## 10.5 강점 4 — DB와 캐싱의 직관

이 챌린지의 핵심 질문: **"무엇을 캐시할 수 있는가?"**

DB의 캐시 적중률을 평소 고민하던 사람은 즉시 답을 떠올린다:
- 변하지 않는 입력 → 캐시.
- TTL, LRU, write-through, cache aside — 모두 똑같이 적용.

7권 캐싱 이론이 이 직관을 AI 추론에 응용한다.

## 10.6 강점 5 — 측정과 검증의 문화

백엔드 엔지니어는 다음을 자연스럽게 한다:
- 변경 전후 측정.
- A/B 테스트.
- 통계적 유의성.
- Regression test.

이게 AI 추론 최적화의 **핵심 엔지니어링 위생**이다 (9권).

## 10.7 약점 인정 — 반대로, 부족한 것들

정직하게:

1. **수학적 직관**: 행렬, 미적분의 "감".
2. **GPU 모델**: CPU와 다른 패러다임.
3. **수치 정밀도**: FP16/FP8의 trade-off.
4. **모델 내부**: Transformer/디퓨전의 흐름.
5. **Python 생태계**: Java/Go 출신에겐 적응 필요.

이 5개가 이 커리큘럼이 메우는 부분이다.

## 10.8 자가 진단

백엔드 출신 강점 5개에 대해 **자기 점수** 를 매겨라 (0~10):

| 강점 | 점수 |
|---|---|
| 시스템 설계 사고 | __ |
| 동시성과 비동기 | __ |
| 운영 사고 | __ |
| DB와 캐싱 직관 | __ |
| 측정과 검증 문화 | __ |

5개 중 3개 이상 7점 이상이면 **AI 추론 엔지니어로의 전환은 강하게 추천**한다.

## 10.9 자가 점검

1. 백엔드 출신의 5가지 강점을 들어라.
2. 그중 본인에게 가장 강한 강점은?
3. 부족한 5가지 중 가장 시급한 것은?
4. 이 챌린지가 백엔드의 어떤 직관을 가장 직접적으로 활용하는가?
5. ML 출신이 약한 부분 중 본인에게 가장 강점인 부분은?

---

# 11장. 학습 경로 결정

## 🎯 이 장의 목적

다음 권을 어디로 갈지, 본인의 상황과 목표에 맞게 **명시적으로 결정**한다.

## 11.1 의사결정 트리

```
                    [당신의 목표는?]
                          │
        ┌────────────────┼────────────────┐
        │                │                │
    챌린지 풀이      전직 준비         탐색 중
    급함            (3개월)           (장기)
        │                │                │
        ▼                ▼                ▼
   속성 코스         표준 코스       마스터 코스
   1, 2, 5, 6,      1~10권         1~11권
   7, 10권 (6권)    (모두)          + 사이드 프로젝트
        │                │                │
        ▼                ▼                ▼
    3주              3개월            6개월
```

## 11.2 코스별 상세

### 11.2.1 속성 코스 (3주)

**목표**: 챌린지 답안 작성에만 집중.

**순서**:
- Volume 1 (지금 권) — 1주
- Volume 2 (디퓨전) — 3일
- Volume 5 (GPU) — 5일
- Volume 6 (최적화) — 5일
- Volume 7 (캐싱) — 3일
- Volume 10 (풀이) — 4일

**스킵하는 권**: 3, 4, 8, 9, 11. 단, 풀이 중 막히면 그때 참조.

### 11.2.2 표준 코스 (3개월)

**목표**: AI 추론 엔지니어 면접에 대응 가능한 수준.

**순서**: 1~10권 순서대로.

**주당 시간**: 10~13시간.

### 11.2.3 마스터 코스 (6개월)

**목표**: 실전 포트폴리오 + 시장 진입.

**순서**: 1~11권 + 사이드 프로젝트 2개.

**사이드 프로젝트 예시**:
- SD/FLUX 모델의 자체 추론 서버 + 벤치마크 리포트.
- LLM의 vLLM 기반 서빙 + KV cache 최적화.

## 11.3 권별 추천 시간 (재확인)

| Vol | 제목 | 추천 시간 |
|---|---|---|
| 1 | 백엔드 → AI 추론 다리 | 6~10h |
| 2 | 디퓨전 기술적 이해 | 8~12h |
| 3 | Transformer 깊이 | 12~16h |
| 4 | FLUX.2 분해 | 10~14h |
| 5 | GPU 시스템 | 14~20h |
| 6 | 추론 최적화 종합 | 16~24h |
| 7 | 캐싱 이론과 실전 | 12~18h |
| 8 | 시스템 설계 | 12~16h |
| 9 | 측정과 검증 | 8~12h |
| 10 | 챌린지 풀이 | 10~16h |
| 11 | 전직 실전 | 지속 |

## 11.4 자가 점검

1. 본인의 목표를 한 문장으로 적어라.
2. 위 3가지 코스 중 어느 코스를 선택하는가?
3. 첫 다음 권을 무엇으로 정했는가?
4. 학습 시간을 주당 몇 시간 확보 가능한가?
5. 첫 자가 마감일(데드라인)을 적어라.

---

# 부록 A — 환경 셋업: Python, PyTorch, CUDA

## A.1 Python 설치

### macOS (Homebrew)

```bash
brew install python@3.11
python3.11 --version  # Python 3.11.x
```

### Ubuntu

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

### Windows

[python.org](https://www.python.org/downloads/) 에서 3.11 다운로드.

## A.2 가상환경

```bash
python3 -m venv venv
source venv/bin/activate          # macOS/Linux
# 또는
venv\Scripts\activate             # Windows
```

## A.3 PyTorch 설치

### CPU only

```bash
pip install torch torchvision
```

### NVIDIA GPU (CUDA 12.x)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

설치 확인:

```python
import torch
print(torch.__version__)              # 2.x.x
print(torch.cuda.is_available())      # True (GPU 있을 때)
print(torch.cuda.get_device_name(0))  # GPU 이름
```

## A.4 추가 라이브러리

```bash
pip install transformers diffusers accelerate jupyter
```

## A.5 권장 IDE

- **VSCode** + Python 확장.
- **PyCharm** Community.
- **Jupyter Lab** (노트북 형태).

---

# 부록 B — 모호한 표현 사전

이 책에서 의도적으로 피한 표현 목록과, 사용한 명확한 표현.

| 모호한 표현 | 이 책의 명확한 표현 |
|---|---|
| "보통은 ~이다" | "X 비율로 ~이다" 또는 "조사 출처: ~" |
| "~할 수도 있다" | "~한다" 또는 "~의 경우 ~한다" |
| "꽤 빠르다" | "X배 빠르다" |
| "많이 쓰인다" | "[프로젝트 X, Y, Z]에서 사용됨" |
| "최근에" | "YYYY년 ~월부터" |
| "거의 모든" | "(특정 제외 사례 명시)" |
| "쉽다" | "(어려운 부분 명시 후) 비교적 쉽다" |
| "복잡하다" | "(어떤 면이 복잡한지 명시)" |

이 사전은 본문 작성 시 self-check로 사용한다.

---

# 1권 마무리

## 🎯 1권의 결산

이 권을 끝낸 당신은:

✅ AI 추론 엔지니어 직무를 명확히 안다.
✅ 백엔드와 AI 추론의 본질이 같다는 자신감을 얻었다.
✅ 행렬·텐서·신경망의 동작을 코드 레벨로 읽을 수 있다.
✅ PyTorch로 첫 추론을 직접 실행했다.
✅ 다음 권을 어디로 갈지 결정했다.

## 다음 권으로

선택에 따라:

- **속성 코스**: → [Volume 2 — 디퓨전 모델의 기술적 이해](./volume_02_diffusion.md) (작성 예정)
- **표준 코스**: → 동일하게 Volume 2
- **마스터 코스**: → 동일하게 Volume 2

> 길은 길지만, 한 권을 끝낸 사람은 두 권을 끝낼 수 있다. 두 권을 끝낸 사람은 모두를 끝낼 수 있다.
>
> 한 걸음씩.
