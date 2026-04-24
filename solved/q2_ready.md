# Q2 Ready — Hancom Docs 자동화 SDK 사전 지식

> 본 문서는 [`task-web.ko.md`](./task-web.ko.md) 풀이에 들어가기 **전에 반드시/권장되는 선행 지식**을 정리한 체크리스트이다.
> 풀이는 아직 시작하지 않으며, **머릿속 모델(mental model)** 을 먼저 세우는 데 사용한다.

---

## A. Chrome DevTools Protocol (CDP)

1. **CDP의 구조**
   - 도메인(Domain) 단위로 묶인 메서드/이벤트.
   - **JSON-RPC over WebSocket**.
2. **타깃 디스커버리**
   - `http://localhost:<port>/json` 으로 타깃 목록 조회.
   - `webSocketDebuggerUrl` 로 연결.
3. **핵심 도메인**
   - `Runtime.evaluate`, `Runtime.callFunctionOn` — **페이지 컨텍스트에서 JS 실행**.
   - `Input.dispatchKeyEvent`, `Input.dispatchMouseEvent`, `Input.insertText` — **키보드·마우스 입력**.
   - `Page`, `DOM`, `Network`, `Debugger`, `Target` 도메인의 역할.
   - `Debugger.setBreakpointByUrl`, `Debugger.paused` — **함수 진입 시 가로채기**.
   - `Page.captureScreenshot` — **시각 검증**용.
4. **`--remote-debugging-port` 와 보안 모델**
   - 브라우저 프로필, 동일 origin 정책.
5. **왜 Puppeteer/Playwright가 아닌 raw CDP인가**
   - 추상화 레이어가 깔리면 **캔버스 에디터 내부에 정확한 키 이벤트 시퀀스를 흘려보내기 어려운** 경우가 있음.

## B. Canvas 기반 에디터의 일반론

6. **Canvas 렌더링 vs DOM 렌더링**
   - Google Docs(2021 canvas 전환), Figma, Office Online 등이 **왜 캔버스로 갔는지**.
   - 선택/렌더링 일관성·성능·교차 브라우저 이슈.
7. **Canvas 에디터의 일반적 내부 구조**
   - **Document Model (in-memory)** — JSON-like tree (paragraphs, runs, tables…).
   - **Layout / Shaping Engine.**
   - **Renderer (canvas 2D 또는 WebGL).**
   - **Input Handler** — 합성 키보드/마우스 → 모델 변경.
   - **Persistence** — 서버와 동기화하는 SDK/REST/WebSocket.
8. **`window.getSelection()` 이 비어있는 이유와 클립보드 차단**
   - 모든 selection이 **캔버스 내부 좌표계**에서만 의미를 가짐.

## C. 리버스 엔지니어링 기법

9. **Chrome DevTools 활용**
   - **Sources 패널** — 번들/소스맵, **Pretty-print, "Search in all files"**.
   - **Performance · Memory snapshot** — 객체 그래프에서 **문서 모델 찾기**.
   - **Network 탭** — 문서 로드 시 받는 **JSON/binary 분석**.
   - **`window` 글로벌 / DevTools console에서 객체 탐색** — **전역에 노출된 모듈 핸들** 찾기.
10. **흔한 진입점**
    - 전역 변수/싱글톤 (예: `window.app`, `window.editor`, `window.__HWP__` 류).
    - **소스맵(.map)** 존재 여부 확인.
    - **WebAssembly 모듈** 사용 여부 (한컴은 hwp 엔진을 wasm으로 컴파일했을 가능성 있음 — **내보내진 함수명 단서**).
    - **`postMessage`, `BroadcastChannel`, `IndexedDB`** 사용 패턴.
11. **함수 후킹 / 몽키패칭**
    - `Runtime.evaluate` 로 원본 함수를 감싸 **호출 인자/리턴값 로깅**.
12. **소스맵 복원, webpack chunk 추적, scope chain 탐색.**

## D. 워드프로세서 도큐먼트 모델 일반론

13. **HWP/HWPX 포맷의 큰 그림**
    - section → paragraph → char run.
    - 표·이미지·스타일 메타데이터.
14. **마크다운 매핑**
    - 헤딩 레벨(스타일명 기반 추론).
    - bold/italic span 합치기.
    - 표/이미지 표현.
15. **OT/CRDT 기반 협업 에디터 구조**
    - 한컴 닥스가 협업을 지원한다면 작업 적용은 **"command" 단위**일 가능성.

## E. SDK 설계 측면

16. **클라이언트 클래스 설계**
    - 연결, 세션 관리, 재연결, 타깃 선택.
17. **비동기 모델**
    - CDP는 비동기 메시지 기반.
    - request/response correlation by `id`.
18. **타이핑 안전성을 위한 wait 전략**
    - 키 입력 후 모델이 업데이트될 때까지 **polling 또는 event listening**.
    - 캔버스라 **DOM 변화가 없음** → 다른 신호 필요.
19. **테스트 전략**
    - 순수 변환 로직 (예: `model JSON → markdown`) 은 **단위 테스트**.
    - CDP 의존부는 **통합/스모크 테스트**.

## F. 운영·디버깅

20. **로그·트레이스**
    - CDP 전후 dump, **재현 가능한 시나리오**.
21. **에러 모드**
    - 페이지 새로고침, 토큰 만료, race condition.

---

## 🎯 핵심 인사이트 방향 (미리 머릿속에 둘 것)

> 챌린지가 "**돌파구가 분명히 존재한다**" 고 명시 → 단순 OCR/픽셀 분석은 **함정**.
>
> 에디터 내부 JS의 **in-memory 도큐먼트 모델 객체** 를
> **전역에서 발견하거나** / **함수 후킹으로 노출**시키는 것이 **정답 라인**일 가능성이 매우 높음.
>
> 그게 되면 **서식·표·검색·내보내기가 한꺼번에** 풀린다.
> 즉, "한 곳을 뚫으면 모든 read 기능이 동시에 해결되는" 지점을 찾는 것이 핵심.

---

## ✅ 풀이 진입 전 자체 체크리스트

- [ ] Chrome을 `--remote-debugging-port=9222` 로 띄우고 `http://localhost:9222/json` 응답을 확인했는가?
- [ ] 한컴 닥스 에디터 페이지에서 **Sources 패널의 번들 파일 목록**을 훑어봤는가?
- [ ] **소스맵(.map) 노출 여부**를 확인했는가?
- [ ] DevTools console에서 `window` 객체의 **non-native 프로퍼티** 들을 살펴봤는가?
- [ ] **WebAssembly 모듈** 로드 여부 / `WebAssembly.Module` 사용 흔적을 확인했는가?
- [ ] Network 탭에서 문서 로드 시 받는 **JSON 또는 바이너리 페이로드**의 구조를 살펴봤는가?
- [ ] **타이핑 한 글자**를 입력했을 때 어떤 함수가 호출되는지 — 콜스택을 잡아봤는가?
- [ ] **저장(Save) 버튼** 클릭 시 발생하는 네트워크 요청을 캡처했는가?
- [ ] 마크다운 변환을 위한 **HWP/HWPX 스키마 → MD 매핑 규칙**의 초안을 그려봤는가?
- [ ] CDP 클라이언트의 **재연결·타임아웃·에러 처리** 정책을 정해뒀는가?

---

## 🧭 실험 권장 순서 (Suggested Exploration Order)

1. **Connect** — raw CDP로 한컴 닥스 탭에 붙기 (`Runtime.evaluate("1+1")` PoC).
2. **Observe** — `window` 글로벌 / Sources / Network / Memory snapshot으로 **에디터 내부 구조**를 1시간 내 매핑.
3. **Hook** — 가장 의심스러운 함수 1~2개를 **몽키패치**해서 호출 데이터 캡처.
4. **Read first** — 가장 쉬운 read부터 (전체 텍스트) → 구조 → 서식 순.
5. **Write second** — `Input.dispatchKeyEvent` 로 타이핑 → 모델 변경 확인.
6. **Compose** — 표 만들기·찾아 바꾸기·저장 등 **고수준 메서드** 조립.
7. **Wrap** — 클래스/메서드 인터페이스 정리, 테스트 작성, 아키텍처 문서 1페이지.
