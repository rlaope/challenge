# 백엔드 엔지니어를 위한 AI 추론 엔지니어 전직 커리큘럼

> **이 챌린지(FLUX.2 try-on 추론 최적화)를 무기로, AI 추론 엔지니어로 전직하기 위한 11권짜리 학습 커리큘럼.**
> 백엔드/서버 개발 경험이 있고, AI는 "ChatGPT 써본 정도"인 개발자를 대상으로 한다.

---

## Part 0 — 시작하기 전에

### 0.1 이 커리큘럼은 누구를 위한 것인가

**대상 독자**:
- Spring/Express/Django/FastAPI 같은 **백엔드 프레임워크 경험 1년 이상**.
- **HTTP, REST, DB, 캐싱, 멀티스레딩, 비동기**의 실전 감각.
- AI/ML은 **chatGPT 사용자, 또는 PyTorch 한두 번 따라 친 정도**.
- 행렬·미적분 기억은 흐릿함 (괜찮음, 다시 다룬다).
- **GPU에 직접 코드 돌려본 적 거의 없음** (괜찮음).

**대상 결과**:
- 이 챌린지(`task-ai.md`)의 답을 **자기 손으로** 작성할 수 있는 수준.
- AI 추론 엔지니어 직무 면접에서 **추론 최적화의 어휘로 대화 가능**.
- 실전 프로젝트로 **포트폴리오 1개 보유**.

### 0.2 학습 로드맵

| 코스 | 기간 | 권 |
|---|---|---|
| **속성 코스 (챌린지 풀이 우선)** | 3주 | 1, 2, 5, 6, 7, 10권 (6권) |
| **표준 코스 (전직 준비 본격)** | 3개월 | 1~10권 |
| **마스터 코스 (포트폴리오까지)** | 6개월 | 1~11권 + 사이드 프로젝트 |

### 0.3 백엔드 → AI 추론 전직의 현실

#### 좋은 소식
- AI 추론 엔지니어 수요는 **매년 폭증**, 공급은 **희소**.
- 백엔드 출신은 **GPU 기반 인프라/시스템 설계 사고**가 강해 차별화됨.
- "**모델 자체를 학습**"하는 ML 엔지니어와 달리, **모델을 빠르게 돌리는** 엔지니어는 시스템적 사고가 핵심.

#### 어려운 소식
- **GPU 프로그래밍**, **저수준 메모리 모델**, **수치 정밀도** 같은 새 어휘 학습 필요.
- **Python + PyTorch** 가 사실상 표준 (Java/Kotlin 출신은 더 많은 적응 필요).
- **수학** 은 끝까지 따라가야 함 (미적분, 선형대수의 직관 수준).

---

## 전체 11권 구성

```
[Volume 1]  백엔드 엔지니어를 위한 AI 추론 입문        ← 다리 놓기
[Volume 2]  디퓨전 모델의 기술적 이해                 ← 모델 패밀리
[Volume 3]  Transformer 아키텍처 깊이                ← 모델 내부
[Volume 4]  FLUX.2 분해 분석                         ← 챌린지 모델
[Volume 5]  GPU 시스템 (백엔드 친화 deep dive)        ← 하드웨어
[Volume 6]  추론 최적화 종합                         ← 가속 카드
[Volume 7]  캐싱 이론과 실전 (백엔드 친화)            ← 핵심 통찰
[Volume 8]  AI 추론 시스템 설계                      ← 서비스 관점
[Volume 9]  측정과 검증                              ← 엔지니어링 위생
[Volume 10] 챌린지 풀이 워크스루                     ← 답안 작성
[Volume 11] 실전 프로젝트와 면접 준비                 ← 포트폴리오
```

---

## Volume 1 — 백엔드 엔지니어를 위한 AI 추론 입문

> **목표:** "AI 추론" 이라는 낯선 영역의 어휘와 사고방식을, 백엔드 개발자의 사고 위에 자연스럽게 얹는다.

### 목차

1. 왜 지금 AI 추론 엔지니어인가 — 시장과 직무 현실
2. 백엔드 vs AI 추론 — 비슷한 점, 다른 점
3. 어휘 변환표 — 백엔드 용어 ↔ AI/ML 용어
4. 행렬·텐서 완전 정복 — DB row/column이 익숙하다면 쉽다
5. 신경망의 본질 — "그냥 거대한 합성 함수다"
6. PyTorch 30분 입문 — Flask 만들 줄 알면 PyTorch도 친숙해진다
7. 첫 모델 추론 — Hugging Face로 한 시간 만에 동작 확인
8. AI 추론 서비스 vs 일반 REST API — 무엇이 다른가
9. 모델의 라이프사이클 — 학습/배포/모니터링
10. 백엔드 엔지니어가 가진 장점 — 시스템 사고는 강한 무기
11. 학습 경로 결정 — 다음에 무엇을 공부할 것인가

### 학습 시간

- 6~10시간 (정독 + 실습).

---

## Volume 2 — 디퓨전 모델의 기술적 이해

> **목표:** "이미지 생성 AI가 어떻게 그림을 만드는지" 의 진짜 동작을 코드 레벨에서 이해.

### 목차

1. 생성 모델의 진화사 — VAE → GAN → Diffusion으로 온 이유
2. DDPM의 수학 — 백엔드 개발자 친화로 풀어쓰기
3. DDIM — 1000 step을 50 step으로 줄인 트릭
4. Latent Diffusion — Stable Diffusion이 폭발한 이유
5. Rectified Flow — 직선 경로로 4 step까지 줄인 다음 세대
6. Distillation — Teacher의 지식을 Student에게 압축
7. CFG (Classifier-Free Guidance) — 매 step 두 번 도는 이유
8. 디퓨전 추론의 한 cycle — 코드 레벨로 따라가기
9. 디퓨전이 LLM과 다른 이유 — KV cache가 안 통하는 이유
10. 실습: Stable Diffusion 1.5 로컬 추론하기

### 학습 시간

- 8~12시간.

---

## Volume 3 — Transformer 아키텍처 깊이

> **목표:** Attention 메커니즘과 Transformer 블록을 처음부터 끝까지. 코드와 수식 모두.

### 목차

1. Transformer 이전 — RNN/LSTM의 한계
2. Attention의 직관 — "관련도를 가중평균"
3. Self-Attention 수식 완전 정복 (Q, K, V)
4. Multi-Head Attention — 왜 머리가 여럿인가
5. Position Encoding — Sinusoidal, Learned, RoPE의 진화
6. Transformer Block 구조 — Attention + FFN + Residual + LayerNorm
7. Encoder vs Decoder — BERT vs GPT의 차이
8. Cross-Attention — 두 시퀀스가 만날 때
9. ViT (Vision Transformer) — 이미지를 패치로 자른다
10. DiT (Diffusion Transformer) — 디퓨전이 Transformer 만나다
11. MM-DiT — FLUX의 더블/싱글 스트림 구조
12. 실습: 100줄 PyTorch로 mini-Transformer 구현

### 학습 시간

- 12~16시간.

---

## Volume 4 — FLUX.2 분해 분석

> **목표:** 챌린지의 모델인 FLUX.2 Klein을 코드 레벨로 완전 해부.

### 목차

1. Black Forest Labs — FLUX 모델 패밀리 한눈에
2. BFL FLUX 코드 베이스 구조 — 디렉토리 투어
3. `model.py` 정독 — Forward pass의 정확한 흐름
4. `autoencoder.py` 정독 — VAE의 인코더/디코더 코드
5. `sampling.py` 정독 — denoising loop의 실제 구현
6. Klein [4B] config 해석 — 각 숫자의 의미
7. Reference image 처리 — 어떻게 토큰이 되는가
8. Attention mask 분석 — 누가 누구를 보는가
9. RoPE position 부여 — 멀티 이미지에서의 까다로운 점
10. 한 forward의 시간 분포 — 어디서 시간을 잡아먹는가
11. 실습: BFL 레포 clone → 코드 함수 호출 그래프 그리기

### 학습 시간

- 10~14시간.

---

## Volume 5 — GPU 시스템 (백엔드 친화 deep dive)

> **목표:** GPU 하드웨어를 백엔드 개발자가 익숙한 개념(스레드 풀, 캐시 계층, 메모리 모델)으로 이해.

### 목차

1. CPU vs GPU — 동시성 모델의 차이 (스레드 풀과 비교)
2. NVIDIA GPU 아키텍처 — SM, Warp, Thread Block
3. CUDA Core vs Tensor Core
4. 메모리 계층 — Register / SRAM / L2 / HBM (Redis L1/L2 cache와 비교)
5. Memory-bound vs Compute-bound — DB query optimization과 비슷
6. Roofline 모델 — 성능 분석 도구
7. 정밀도 — FP32 vs FP16 vs BF16 vs FP8 vs FP4
8. Hopper (H100) 아키텍처 — TMA, WGMMA, FP8
9. Ada (4090) 아키텍처 — 컨슈머급 강자
10. CUDA 프로그래밍 — 백엔드의 멀티스레딩과 비교
11. cuDNN, cuBLAS, CUTLASS — 표준 라이브러리들
12. Triton — Python으로 GPU 커널 짜기
13. 실습: torch.profiler로 모델 시간 분석

### 학습 시간

- 14~20시간.

---

## Volume 6 — 추론 최적화 종합

> **목표:** GPU 가속 카드들을 모두 모아서 조합하는 법.

### 목차

1. Operator Fusion — 여러 작은 연산을 하나로 합치기
2. FlashAttention 1 — 표준 attention의 메모리 IO 문제 해결
3. FlashAttention 2 — Work partitioning 개선
4. FlashAttention 3 — Hopper의 TMA/WGMMA 활용
5. torch.compile — PyTorch 2.0의 자동 컴파일러
6. CUDA Graphs — Python 오버헤드 제거
7. TensorRT — 추론 컴파일러
8. TensorRT-LLM — LLM 특화
9. TransformerEngine — FP8 자동 관리
10. Quantization 종합 — FP8, INT8, INT4, AWQ, GPTQ
11. Paged KV Cache (vLLM) — LLM 추론 인프라
12. 실습: SDXL을 TensorRT로 컴파일해서 속도 측정

### 학습 시간

- 16~24시간.

---

## Volume 7 — 캐싱 이론과 실전 (백엔드 친화)

> **목표:** 백엔드의 캐싱 직관을 AI 추론에 응용. 이 챌린지의 핵심 통찰이 되는 부.

### 목차

1. 캐싱의 본질 — Redis/Memcached에서 시작
2. Cache 적중률 vs 일관성 — 백엔드 캐싱과 같은 트레이드오프
3. KV Cache (LLM 추론) — autoregressive와 캐싱의 자연스러운 만남
4. 디퓨전이 KV cache 못 쓰는 이유 — autoregressive가 아니다
5. Step-wise feature caching — DeepCache, Learning-to-Cache
6. ∆-DiT, FORA, TeaCache — 차분 기반 캐싱 패밀리
7. Reference token K/V caching — **이 챌린지의 핵심**
8. VAE encode 캐싱 — 입력의 변하지 않는 부분
9. 캐시 갱신 정책 — TTL, LRU, 사용자 세션 기반
10. 분산 캐시 — 여러 GPU에 걸친 캐싱
11. 실습: SD에 DeepCache 적용해서 속도/품질 측정

### 학습 시간

- 12~18시간.

---

## Volume 8 — AI 추론 시스템 설계

> **목표:** 모델 가속만으로 끝나는 게 아니라, 서비스 전체의 시스템 설계 시야.

### 목차

1. AI 추론 서비스 아키텍처 — Triton, vLLM, TGI
2. Cold start vs Warm start — 모델 로딩과 GPU 워밍업
3. Continuous batching — vLLM의 핵심 아이디어
4. Sticky session — 같은 사용자는 같은 GPU로
5. Auto-scaling — GPU의 비싼 cold start를 어떻게
6. Multi-tenancy — 여러 모델이 한 GPU에 공존
7. Latency vs Throughput trade-off
8. SLA 설계 — p50/p95/p99
9. 모니터링 — 모델 품질 회귀, 분포 drift
10. 비용 최적화 — H100 vs A100 vs L40S vs RTX 4090
11. 실습: FastAPI로 SD 추론 서버 띄워서 부하 테스트

### 학습 시간

- 12~16시간.

---

## Volume 9 — 측정과 검증

> **목표:** 빠르게 만들었다고 끝이 아니다. 정확한 측정과 품질 보장의 엔지니어링.

### 목차

1. 벤치마크 위생 — warmup, sync, isolation
2. 측정 도구 — torch.profiler, NSight Systems, NSight Compute
3. 시간 분포 분석 — 어디서 진짜 시간이 가는가
4. 품질 메트릭 — LPIPS, FID, CLIP-similarity, SSIM, PSNR
5. A/B 테스트 — 두 최적화 비교
6. Regression test — 품질 회귀 자동 감지
7. Canary deployment — 점진적 롤아웃
8. 통계적 유의성 — N번 측정의 분산
9. Roofline 분석 — 한계까지 갔는가
10. 실습: 한 모델에 대한 종합 벤치마크 리포트 작성

### 학습 시간

- 8~12시간.

---

## Volume 10 — 챌린지 풀이 워크스루

> **목표:** 이 챌린지(`task-ai.md`)의 답안을 직접 작성. 1~9권의 종합.

### 목차

1. Part 1 답안 작성 — 아키텍처 분석과 입력 설계
   - 1-1. VAE 인코더 동작 분석
   - 1-2. Reference/noise/text 토큰 조립과 attention 규칙
   - 1-3. Reference 토큰의 최적화 가능성
   - 1-4. 아바타+의류 입력 전략 제안
2. Part 2 답안 작성 — 최적화 전략
   - 2-1. VAE-side reuse (아바타 latent caching)
   - 2-2. Transformer-side reuse (reference K/V caching)
   - 2-3. Step-wise caching (Learning-to-Cache)
   - 2-4. FlashAttention 3 + FP8
   - 2-5. torch.compile + CUDA Graph
   - 2-6. TensorRT 통합
3. Part 3 답안 작성 — 실험 계획
   - 3-1. 메트릭 선정
   - 3-2. 우선순위와 순서
   - 3-3. 품질 회귀 테스트
4. 최종 문서 정리 — 추론의 깊이를 보여주기

### 학습 시간

- 10~16시간 (실제 답안 작성).

---

## Volume 11 — 실전 프로젝트와 면접 준비

> **목표:** 포트폴리오와 시장 진입.

### 목차

1. 사이드 프로젝트 아이디어 5선
2. GitHub 포트폴리오 구성 — README, 벤치마크, 데모
3. 블로그 작성 — 기술 글쓰기로 신뢰 구축
4. 이력서 — 백엔드 → AI 추론 어떻게 표현
5. AI 추론 엔지니어 직무 시장 — 회사별 특징
6. 면접 단골 질문 — 시스템 설계, 최적화, 트러블슈팅
7. 코딩 인터뷰 — Python/PyTorch 위주
8. 실전 인터뷰 모의

### 학습 시간

- 지속적 (1~3개월).

---

## 권별 의존도 그래프

```
Volume 1 (다리)
    ↓
Volume 2 (디퓨전)  Volume 3 (Transformer)
       ↓                  ↓
       └───→ Volume 4 (FLUX) ←┘
                ↓
Volume 5 (GPU)  Volume 7 (캐싱)
       ↓             ↓
       └─→ Volume 6 (가속) ←─┘
                ↓
       Volume 8 (시스템)  Volume 9 (측정)
                ↓             ↓
                └─→ Volume 10 (풀이) ←─┘
                        ↓
                Volume 11 (전직 실전)
```

---

## 권별 진척 체크리스트

| Vol | 제목 | 시간 | 상태 |
|---|---|---|---|
| 1 | 백엔드 → AI 추론 다리 | 6~10h | ⬜ |
| 2 | 디퓨전 기술적 이해 | 8~12h | ⬜ |
| 3 | Transformer 깊이 | 12~16h | ⬜ |
| 4 | FLUX.2 분해 | 10~14h | ⬜ |
| 5 | GPU 시스템 | 14~20h | ⬜ |
| 6 | 추론 최적화 종합 | 16~24h | ⬜ |
| 7 | 캐싱 이론과 실전 | 12~18h | ⬜ |
| 8 | 시스템 설계 | 12~16h | ⬜ |
| 9 | 측정과 검증 | 8~12h | ⬜ |
| 10 | 챌린지 풀이 | 10~16h | ⬜ |
| 11 | 전직 실전 | 지속 | ⬜ |

**총 학습 시간**: 약 110~160시간 (3개월 기준 주당 10~13시간).

---

## 학습 원칙

### 1. 매 권 시작 전: 명확한 목표 세팅
> "이 권을 끝내면 무엇을 할 수 있어야 하는가?" 를 한 줄로 적기.

### 2. 매 장 끝: 자기 설명 시험
> 책을 덮고, 그 장의 내용을 **백지에 자기 말로** 다시 써보기.
> 못 쓰는 부분이 있다면 그 부분을 다시 읽기.

### 3. 매 권 끝: 코드 실습 1개
> 이론만으로는 휘발됨. **반드시 손으로 친 코드** 한 줄이 있어야 한다.

### 4. 막히면 바로 검색 → 그 다음 AI에게 묻기
> 검색이 더 빠를 때가 많다.
> AI는 "이 개념의 다른 비유를 알려줘" 같은 질문에 강하다.

### 5. 조급해하지 않기
> 백엔드도 1년에 못 익혔다. AI 추론도 그렇다.
> 매주 측정 가능한 진척이 있으면 충분하다.

---

## 다음 단계

지금부터 [`volume_01_backend_bridge.md`](./volume_01_backend_bridge.md) 를 펼친다.

**1권부터 순서대로** 읽어도 좋고, 챌린지 답안 작성이 급하면 **속성 코스(1, 2, 5, 6, 7, 10권)** 로 가도 된다.

행운을 빈다. 이 길은 길지만, 끝에 도달한 사람은 충분히 보상받는다.
