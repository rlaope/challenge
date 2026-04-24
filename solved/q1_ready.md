# Q1 Ready — FLUX.2 추론 최적화 사전 지식

> 본 문서는 [`task-ai.ko.md`](./task-ai.ko.md) 풀이에 들어가기 **전에 반드시/권장되는 선행 지식**을 정리한 체크리스트이다.
> 풀이는 아직 시작하지 않으며, **머릿속 모델(mental model)** 을 먼저 세우는 데 사용한다.

---

## A. Diffusion / Flow 모델의 기초

1. **Diffusion Model vs Rectified Flow**
   - DDPM/DDIM과 비교했을 때 Rectified Flow(RF)가 무엇이 다른가.
   - **직선 경로(straight trajectory)** 로 인해 적은 step으로 sampling이 가능한 이유.
2. **Distillation**
   - "4 step distilled" 모델이 무엇이고, 일반 50-step diffusion과 **품질·속도 트레이드오프**가 어떻게 다른가.
3. **Classifier-Free Guidance (CFG)**
   - 일반적으로 batch가 2배가 되는 이유.
   - Klein은 `guidance_scale=1.0` 으로 **CFG가 없다**는 것의 의미 (= 한 forward만 돌리면 됨).

## B. FLUX.2 / DiT 아키텍처

4. **Diffusion Transformer (DiT)**
   - UNet 기반 대신 **Transformer 기반 디퓨전**이 어떻게 구성되는지.
5. **Double-stream / Single-stream block**
   - FLUX 특유의 **MM-DiT** 구조 (text/image 토큰을 분리해 처리하다 합치는 방식).
6. **Patchify / 2×2 patch embedding**
   - 이미지 latent를 토큰화하는 방식.
   - `in_channels = 128 = 32 z-channels × 2×2 patch` 의 의미.
7. **RoPE (Rotary Positional Embedding) — 2D 버전**
   - 이미지 토큰의 **좌표 인코딩** 방식.
   - 캐싱·재사용 시 **좌표 충돌(positional ID 재할당)** 이 핵심 이슈.
8. **Reference image / context injection**
   - 텍스트·노이즈·레퍼런스 토큰이 **시퀀스에 어떻게 concat** 되는지.
   - **attention mask** 가 어떻게 걸리는지 (full attention vs masked).

## C. VAE

9. **VAE Encoder/Decoder 구조**
    - Conv 기반 다운샘플링, latent space 차원, "이미지 → latent" 비율 (보통 8×).
10. **Convolution의 지역성(locality)**
    - 입력 이미지의 **일부만 바뀌면 latent의 일부만** 바뀐다는 사실.
    - = **부분 캐싱·tiled VAE** 가능성.
11. **VAE의 receptive field**
    - 어디까지가 **독립적으로 인코딩 가능**한지의 경계.

## D. Transformer 추론 최적화 일반

12. **KV Cache**
    - Causal LM에서의 KV 캐시 개념.
    - **diffusion에서는 왜 일반적인 KV cache가 안 되는지** (모든 토큰이 매 step마다 바뀜).
13. **Cross-attention / Self-attention 분리 여부**
    - 어떤 부분이 step 간에 변하지 않는지.
14. **Step-wise feature caching**
    - DeepCache, **Learning-to-Cache**, ∆-DiT, FORA, TeaCache 등.
    - denoising step 사이에 transformer 블록 출력을 재사용하는 기법 패밀리.
    - 챌린지가 **명시적으로 권장**하는 방향.
15. **Token pruning / merging**
    - 레퍼런스 토큰 일부를 **줄이거나 합치는** 기법.

## E. GPU / 커널 / 정밀도

16. **FP16 / BF16 / FP8 (E4M3, E5M2)** 차이.
    - **Hopper(SM90)의 FP8 Tensor Core**, Ada(SM89)의 FP8 지원 한계.
17. **FlashAttention 2 / 3**
    - **FlashAttention-3 on Hopper** (warpgroup MMA, async).
    - 어텐션의 **메모리 대역폭·속도 개선 원리**.
18. **NVIDIA 추론 스택 구성 요소**
    - **TensorRT / TensorRT-LLM, TransformerEngine, cuDNN, CUTLASS, Triton** — 각자의 역할.
19. **torch.compile / CUDA Graphs**
    - **정적 그래프화로 Python overhead 제거**.
20. **Quantization 종류**
    - Weight-only(int8/int4), W8A8, **SmoothQuant, AWQ, GPTQ**, NVFP4(블랙웰) 정도까지.

## F. 시스템·서비스 관점

21. **Cold start vs warm start**
    - **KV/feature cache의 세션 단위 보관**.
    - "아바타는 세션 동안 고정"이라는 **도메인 인사이트** 가 시스템 설계에 어떻게 들어가는가.
22. **Latency vs Throughput, p50/p95**
    - **batch=1 추론** 의 특성.
23. **Benchmark 위생**
    - warmup, GPU sync (`torch.cuda.synchronize`), **캐시 격리**.
    - 품질 메트릭(LPIPS, FID, CLIP-similarity, SSIM) 등.

---

## 🎯 핵심 인사이트 방향 (미리 머릿속에 둘 것)

> 아바타는 세션 동안 **불변(invariant)** → 아바타에 해당하는
> (1) **VAE encode 결과**,
> (2) **transformer 내부의 reference-token 관련 연산**(특히 self-attention의 K/V projection)
> 을 **클릭 사이에 캐시**할 수 있는가가 이 과제의 **큰 축**으로 보임.
>
> 또한 챌린지는 **"VAE-side 재사용 + Transformer-side 재사용을 모두 다룬 답"** 을 원하므로,
> 한쪽으로만 치우친 답은 감점 요인.

---

## ✅ 풀이 진입 전 자체 체크리스트

- [ ] BFL flux 레퍼런스 코드(`model.py`, `sampling.py`, `autoencoder.py`)를 직접 열어보고 **호출 그래프**를 그려봤는가?
- [ ] **두 개 이상의 denoising path** 가 존재한다는 힌트를 코드에서 확인했는가?
- [ ] VAE 인코더의 **다운샘플링 단계** 와 receptive field를 손으로 계산해봤는가?
- [ ] 레퍼런스 토큰의 **포지셔널 인코딩** 이 어디서 부여되는지 확인했는가?
- [ ] **DiT caching 논문**(Learning-to-Cache 외 1~2편) 을 훑어 캐싱 단위(블록/스텝/채널)를 비교해봤는가?
- [ ] **NVIDIA 측 공식 FLUX 최적화 레포/블로그** (TensorRT 디퓨전 가이드 등) 를 검색해봤는가?
- [ ] 측정·검증을 위한 **품질 지표** 를 사전에 정해뒀는가?
