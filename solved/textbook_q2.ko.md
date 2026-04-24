# 『Hancom Docs 자동화 SDK를 위한 개념 정리서』
> Q2 챌린지를 풀기 위한 모든 개념을, **공부용 책의 호흡**으로, **쉬운 예시 + 왜 필요한가 + 옛날엔 어땠나 + 어떻게 해결했나** 의 4박자로 풀어 쓴 학습서.

---

## 머리말

이 책은 빠르게 읽고 잊어버리는 글이 아니다. **이 챌린지를 풀러 들어가는 사람을 위한 입문서**이며,
각 개념마다 "왜 세상이 이걸 필요로 했는지"부터 시작한다.

Q2는 Q1과 달리 **눈에 안 보이는 시스템 안을 손전등으로 비춰가며 길을 찾는** 일이다.
보이지 않는 것을 다루기 때문에, **개념의 역사적 맥락**을 알면 어디를 비춰야 할지 직관이 생긴다.

각 장의 구조는 다음과 같다.

> **🎯 한 줄 정의 → 🎈 쉬운 비유 → 🤔 왜 필요한가 → 😣 옛날엔 무엇이 불편했나 → 💡 어떻게 해결했는가 → 🔬 더 깊이**

---

# 제1부 — Chrome DevTools Protocol (CDP)

## 1장. CDP의 구조 — 도메인과 JSON-RPC

### 🎯 한 줄 정의

**Chrome 브라우저를 외부에서 조종할 수 있게 해주는 통신 규약.** 도메인(Domain) 단위로 묶인 메서드/이벤트의 집합이며, JSON-RPC 형식의 메시지를 WebSocket으로 주고받는다.

### 🎈 쉬운 비유

> 자동차에 **OBD-II 진단 포트** 를 꽂으면 외부 컴퓨터로 차의 상태를 읽고 명령을 내릴 수 있다.
> 엔진 도메인, 변속기 도메인, 브레이크 도메인 등 **부품별로 명령이 정리** 돼 있다.
>
> CDP가 그렇다. Chrome에 꽂는 진단 포트.
> - **Page 도메인**: 페이지 로드, 스크린샷.
> - **Runtime 도메인**: JavaScript 실행.
> - **Input 도메인**: 마우스/키보드 이벤트.
> - **DOM 도메인**: HTML 트리 조작.
> - 등등.

### 🤔 왜 필요한가

브라우저 안에서 일어나는 일을 **외부에서 자동화** 해야 하는 경우가 많다:
- 자동화 테스트 (E2E).
- 웹 크롤링.
- 디버깅 도구.
- AI 에이전트 (브라우저 사용).

브라우저가 **표준화된 외부 인터페이스** 를 제공하지 않으면 매번 직접 만들어야 함.

### 😣 옛날엔 무엇이 불편했나

옛날에는 브라우저 자동화가 **각 브라우저마다 다른 방식** 이었다:
- IE: COM 자동화.
- Firefox: Marionette.
- Selenium WebDriver: 표준이긴 했지만 **느리고 한정적**.

Selenium은:
- 브라우저 → WebDriver → 테스트 코드의 **여러 레이어** 를 거침.
- 각 레이어마다 오버헤드.
- 디버깅이나 깊은 조작이 어려움.

### 💡 어떻게 해결했는가

**Chrome DevTools Protocol** (Google, 2015~):
- Chrome이 **DevTools(개발자 도구) 자체** 와 통신하기 위해 만든 프로토콜을 외부에 공개.
- DevTools가 할 수 있는 모든 것을 외부 코드도 할 수 있게 됨.
- **"DevTools가 곧 자동화"** 의 사상.

특징:
- **JSON-RPC over WebSocket**: 요청-응답이 JSON 메시지.
- **양방향**: 브라우저가 이벤트를 푸시 (페이지 로드 완료 등).
- **Domain 단위로 명령 정리** → 발견 가능성 좋음.

### 🔬 더 깊이 — JSON-RPC 메시지 형태

요청:
```json
{
  "id": 1,
  "method": "Runtime.evaluate",
  "params": {
    "expression": "1 + 1",
    "returnByValue": true
  }
}
```

응답:
```json
{
  "id": 1,
  "result": {
    "result": { "type": "number", "value": 2 }
  }
}
```

이벤트(요청 없이 브라우저가 보냄):
```json
{
  "method": "Page.loadEventFired",
  "params": { "timestamp": 12345 }
}
```

`id` 로 요청-응답을 매칭한다 (비동기).

---

## 2장. 타깃 디스커버리 — `/json` 엔드포인트와 WebSocket URL

### 🎯 한 줄 정의

**브라우저에 어떤 탭/프로세스가 있는지 목록을 받아오고, 그중 특정 타깃에 WebSocket으로 연결하는 절차.**

### 🎈 쉬운 비유

> 호텔 프런트에 가서 "오늘 묵는 손님 명단 주세요" 라고 요청한다.
> 명단을 받으면 그중 한 명의 방으로 직접 전화를 건다.
>
> CDP에서:
> - "프런트" = `http://localhost:9222/json`
> - "명단" = 열려 있는 탭/페이지/워커 목록.
> - "방으로 전화" = 각 항목의 `webSocketDebuggerUrl` 로 WebSocket 연결.

### 🤔 왜 필요한가

브라우저에는 보통 **여러 탭** 이 열려 있다. 어디에 연결할지 골라야 한다.
- 한컴 닥스 탭 vs 유튜브 탭 vs 새 탭.
- 각 탭은 **독립된 JavaScript 컨텍스트**.

### 😣 옛날엔 무엇이 불편했나

전통적인 자동화는 **새 브라우저를 띄워서** 그 브라우저만 조종.
- **이미 사용자가 쓰고 있는 브라우저** 에 붙기 어려움.
- 사용자의 로그인 세션을 활용 못 함.
- 한컴 닥스처럼 **수동 로그인이 필요한 SaaS** 에 자동화 적용 어려움.

### 💡 어떻게 해결했는가

**Remote Debugging 모드**:
- Chrome을 `--remote-debugging-port=9222` 로 실행.
- 또는 `chrome://inspect/#remote-debugging` 에서 켜기.
- 그러면 Chrome이 `localhost:9222` 에 HTTP/WebSocket 서버를 연다.

**디스커버리 흐름**:
1. `GET http://localhost:9222/json/version` → 브라우저 정보.
2. `GET http://localhost:9222/json` → 타깃 목록.
3. 각 항목에 `webSocketDebuggerUrl` 가 있음.
4. 그 URL로 WebSocket 연결.
5. JSON-RPC 메시지 송수신 시작.

### 🔬 타깃 종류

- **page**: 일반 웹 페이지.
- **iframe**: 페이지 안의 iframe.
- **worker / service_worker**: 백그라운드 워커.
- **shared_worker**: 공유 워커.
- **browser**: 브라우저 자체 (탭이 아닌 전역 명령용).

### 🔥 이 챌린지에서

- 한컴 닥스 탭(`webhwp.hancomdocs.com`)을 **제목/URL 매칭** 으로 찾아 연결.
- iframe 안에 에디터가 있을 수도 있으므로 iframe 타깃도 살펴야 함.
- Worker 안에 핵심 로직이 있을 수도 있음.

---

## 3장. 핵심 도메인 — Runtime / Input / Page / DOM / Debugger / Target

### 🎯 한 줄 정의

CDP의 명령들이 **기능별로 묶여있는 그룹.** 각 도메인은 독립적인 명령/이벤트 모음.

### 🎈 쉬운 비유

> 큰 회사의 **부서** 같다.
> - **Runtime 부서**: JS 실행 담당.
> - **Input 부서**: 키보드/마우스 담당.
> - **Page 부서**: 페이지 로드/스크린샷 담당.
> - **Network 부서**: 네트워크 요청 가로채기.
> - **Debugger 부서**: 중단점/스텝 실행.

### 🤔 왜 필요한가

CDP는 명령이 **수백 개** 있다. 그룹핑이 없으면 찾기 어렵다.

### 😣 옛날엔 무엇이 불편했나

평면적인 API였다면:
- `evaluate`, `dispatchKey`, `captureScreenshot`, … 모두 한 통.
- 충돌, 발견 어려움.

### 💡 어떻게 해결했는가

**도메인 네임스페이스**:
- `Runtime.evaluate` (Runtime 도메인의 evaluate).
- `Input.dispatchKeyEvent`.
- `Page.captureScreenshot`.
- `DOM.getDocument`.

도메인별 문서가 분리됨 → 각 도메인 안에서만 탐색하면 됨.

### 🔬 도메인별 핵심 명령들 — 이 챌린지에 필요한 것

#### Runtime 도메인 (가장 중요)
- `Runtime.evaluate(expression)` — 페이지 컨텍스트에서 JS 실행.
- `Runtime.callFunctionOn(objectId, functionDeclaration)` — 특정 객체에 함수 호출.
- `Runtime.getProperties(objectId)` — 객체의 속성 목록 (디버깅에 유용).
- `Runtime.consoleAPICalled` (이벤트) — `console.log` 등의 출력 받기.

#### Input 도메인 (쓰기 작업의 핵심)
- `Input.dispatchKeyEvent({type: 'keyDown', key: 'A', ...})` — 키 이벤트.
- `Input.insertText(text)` — 텍스트 삽입 (IME 대응).
- `Input.dispatchMouseEvent` — 마우스 이벤트.

#### Page 도메인
- `Page.navigate(url)` — 페이지 이동.
- `Page.captureScreenshot()` — 스크린샷 (시각 검증용).
- `Page.frameStartedLoading` (이벤트) — 페이지 변화 감지.

#### DOM 도메인
- `DOM.getDocument()` — DOM 트리.
- `DOM.querySelector(nodeId, selector)` — 셀렉터로 검색.
- 캔버스 에디터에는 별 도움 안 되지만, 메뉴/버튼 같은 외곽은 DOM.

#### Debugger 도메인 (리버스 엔지니어링의 핵심 무기)
- `Debugger.enable()` — 디버거 켜기.
- `Debugger.setBreakpointByUrl(url, lineNumber)` — 특정 위치에 중단점.
- `Debugger.paused` (이벤트) — 중단점 도달 시 발생, **호출 스택과 변수** 받을 수 있음.
- `Debugger.evaluateOnCallFrame` — 중단된 컨텍스트에서 변수 평가.

#### Target 도메인
- `Target.attachToTarget` — 다른 타깃(iframe 등)에 attach.
- `Target.setDiscoverTargets(true)` — 새 타깃 자동 발견.

---

## 4장. `--remote-debugging-port` 와 보안 모델

### 🎯 한 줄 정의

**Chrome을 원격 제어할 수 있게 여는 명령줄 옵션.** 보안상 매우 강력해서 함부로 켜면 안 된다.

### 🎈 쉬운 비유

> 집의 **현관문 열쇠** 를 외부 사람에게 맡기는 것과 같다.
> 잘 쓰면 편리한 자동화, 잘못 쓰면 침입.
>
> 그래서 Chrome은:
> - 기본적으로 꺼져 있음.
> - localhost에서만 접근 가능 (네트워크 외부에서 직접 연결 불가).

### 🤔 왜 필요한가

Remote debugging이 켜져 있어야:
- 외부 코드가 Chrome에 접속.
- 자동화 가능.

### 😣 옛날엔 무엇이 불편했나

처음에는 "remote debugging" 이 너무 강력해서:
- 키 입력 가로채기.
- 패스워드 노출.
- 사용자의 로그인 세션 도용.
- → 보안 사고 사례 발생.

### 💡 어떻게 해결했는가

**Chrome의 안전장치들**:
1. **localhost only** (기본): 외부에서 직접 못 붙음.
2. **`--remote-debugging-address=0.0.0.0` 차단** (최근): 외부 노출 어렵게.
3. **별도 user data dir 권장**: `--user-data-dir=/tmp/chrome-debug` — 평소 쓰는 프로필과 분리.
4. **--remote-allow-origins**: 어느 origin에서 WS 접근 허용할지 명시.

### 🔬 이 챌린지에서

- 사용자가 직접 Chrome을 띄움 → SDK는 그 인스턴스에 붙음.
- 보안 책임은 사용자에게.
- 단, **별도 프로필 사용 권장** (한컴 로그인 세션과 평소 사용자 세션 분리).

---

## 5장. 왜 raw CDP 인가? (vs Puppeteer / Playwright)

### 🎯 한 줄 정의

**Puppeteer/Playwright는 CDP 위에 만들어진 고수준 라이브러리.** 편리하지만 추상화가 끼면 캔버스 에디터의 미묘한 동작을 정확히 다루기 어려울 수 있다.

### 🎈 쉬운 비유

> 자동차 운전에 비유하면:
> - **자동변속(Puppeteer)**: 편하다. 보통의 운전엔 충분.
> - **수동변속(raw CDP)**: 손이 더 가지만, **특수한 상황** (드리프트, 산악 운전) 에서 정확한 제어 가능.
>
> 한컴 닥스 같은 **특수한 캔버스 에디터** 는 보통 패턴을 벗어남.
> → raw CDP가 더 유리할 수 있음.

### 🤔 왜 필요한가

Puppeteer/Playwright는 너무 많은 걸 자동으로 처리한다:
- "키 입력 후 페이지 변화 기다리기" 같은 wait 로직이 **DOM 기반**.
- **DOM이 안 변하는 캔버스 에디터** 에서는 wait가 무한 대기.
- 또는 잘못된 상태에서 다음 동작 진행.

### 😣 옛날엔 무엇이 불편했나

Puppeteer로 복잡한 캔버스 앱(Figma, 게임)을 자동화하려다 실패한 사례 다수.
- 추상화 우회를 위해 결국 raw CDP로 갈아탐.

### 💡 어떻게 해결했는가

**raw CDP 직접 사용**:
- WebSocket 라이브러리만 있으면 됨 (`ws` for Node.js, `websockets` for Python).
- JSON-RPC 메시지 직접 송수신.
- **모든 동작을 명시적으로** 제어.
- wait는 **에디터 내부 상태** 를 직접 관찰해서 결정.

### 🔬 단점

- 코드량이 많아짐.
- 재연결 / 에러 처리 직접 짜야 함.
- → 그래서 챌린지에서 "**클라이언트 클래스로 잘 감싸라**" 라고 한 것.

### 🔥 이 챌린지의 권장 사항

- **raw CDP 사용**.
- 단, 깔끔한 클래스로 감싸서 **이 챌린지의 SDK가 곧 추상화 레이어** 가 되도록.

---

# 제2부 — Canvas 기반 에디터의 일반론

## 6장. Canvas 렌더링 vs DOM 렌더링

### 🎯 한 줄 정의

- **DOM 렌더링**: HTML 요소(div, span 등)를 사용해 콘텐츠 표현.
- **Canvas 렌더링**: 모든 콘텐츠를 픽셀로 그림.

### 🎈 쉬운 비유

> 미술 시간:
> - **DOM**: 색종이를 오려 붙이기 (각 조각이 독립적인 요소).
> - **Canvas**: 도화지에 그림 그리기 (모든 게 픽셀, 분리 불가).

### 🤔 왜 필요한가

웹 에디터들이 캔버스로 옮긴 이유:
- **렌더링 일관성**: 모든 브라우저에서 정확히 같은 모습.
- **성능**: 큰 문서에서 DOM은 느림 (수만 개 노드 시 죽음).
- **세밀한 제어**: 폰트, 자간, 줄바꿈을 픽셀 단위로 제어.

### 😣 옛날엔 무엇이 불편했나

Google Docs (2010년대 초)는 DOM 기반이었다.
- 큰 문서에서 느림.
- 폰트 렌더링이 브라우저마다 다름.
- 협업 시 커서 동기화가 어려움.

→ **2021년 Google Docs가 캔버스로 전환**.
같은 흐름: Figma, Office Online, Notion(부분), 그리고 한컴 닥스.

### 💡 어떻게 해결했는가

**캔버스 에디터의 일반 구조**:
```
[Document Model (메모리 안 JSON 트리)]
         ↓
[Layout Engine (글자 배치 계산)]
         ↓
[Renderer (canvas에 픽셀 그리기)]
         ↓
[Canvas (사용자가 보는 화면)]
```

사용자가 키를 누르면 → Document Model 변경 → 다시 Layout → 다시 Render.

### 🔬 자동화 관점에서의 함의

- **DOM에는 텍스트가 없다** → DOM 스크래핑 무용.
- 진짜 데이터는 **JavaScript 메모리 안의 Document Model 객체**.
- → **그 객체를 찾아내는 게 자동화의 핵심**.

이게 챌린지의 출발점.

---

## 7장. Canvas 에디터의 내부 구조 (5요소)

### 🎯 한 줄 정의

캔버스 에디터는 보통 **(1) Document Model, (2) Layout, (3) Renderer, (4) Input Handler, (5) Persistence** 의 5층 구조.

### 🎈 쉬운 비유

> **연극 무대** 로 비유:
> 1. **Document Model** = **대본** (실제 내용).
> 2. **Layout** = **연출** (누가 어디 서서 무엇을 할지).
> 3. **Renderer** = **무대 위 배우들의 실제 동작** (관객이 보는 것).
> 4. **Input Handler** = **연출자의 지시** (대본을 어떻게 바꿀지 받음).
> 5. **Persistence** = **공연이 끝나고 대본을 출판** (서버 저장).

### 🤔 왜 이 구조인가

분리하면:
- 각 층을 독립적으로 최적화 가능.
- 유지보수 쉬움.
- 협업/Undo/Redo 구현 쉬움 (Document Model 위주).

### 😣 옛날엔 무엇이 불편했나

옛날엔 **렌더링과 데이터가 섞여 있었다** (jQuery 시대 코드).
- 한 줄 바꾸려면 여러 곳 수정.
- 협업 구현 거의 불가능.

### 💡 어떻게 해결했는가

**MVC / MVVM / Flux 패턴** 의 적용:
- Model (data) ↔ View (display) 분리.
- 단방향 데이터 흐름.
- React, Redux 등이 표준화.

캔버스 에디터에 응용:
- Document Model은 순수 JSON.
- Renderer는 Model을 받아 그리기만 함.
- Input은 Model을 변경하는 명령 (Command).

### 🔬 각 층의 자동화 접근법

| 층 | 자동화 방법 |
|---|---|
| **Document Model** | 가장 좋은 진입점. **객체를 찾아내면 read 다 풀림** |
| **Layout** | 거의 안 건드림 |
| **Renderer** | 시각 검증용 (스크린샷) |
| **Input Handler** | 쓰기 작업 시 호출 (키 이벤트 → handler) |
| **Persistence** | save 트리거 / 변경 감지 |

> **챌린지의 "breakthrough"**: Document Model 객체를 어떻게 손에 넣느냐.

---

## 8장. `window.getSelection()` 이 비어있는 이유와 클립보드 차단

### 🎯 한 줄 정의

**캔버스 에디터에서는 브라우저의 표준 selection/clipboard API가 동작하지 않는다.** 모든 selection이 캔버스 내부 좌표계에서만 의미를 가지기 때문.

### 🎈 쉬운 비유

> 그림 위에 글자를 **그림으로** 그려놨다.
> 마우스로 글자를 드래그해도 **그건 그림이지 글자가 아니다**.
> 브라우저는 글자가 어디에 있는지 모르니, "선택된 텍스트" 가 없다.
>
> 그래서 `window.getSelection()` 이 빈 문자열을 반환한다.

### 🤔 왜 이게 문제인가

표준 자동화는 보통:
- "선택해서 복사" → `Ctrl+A, Ctrl+C` → 클립보드 읽기.

캔버스 에디터에서는:
- `Ctrl+A` 는 **에디터 내부에서만** 작동 (DOM selection 아님).
- `Ctrl+C` 도 **에디터 내부 핸들러** 가 처리. 표준 클립보드에 안 들어감.
- 또는 보안상 **클립보드 API가 막힘** (HTTPS + 사용자 제스처 필요).

### 😣 옛날엔 무엇이 불편했나

옛날 (DOM 에디터) 엔:
- `Ctrl+A` → DOM selection.
- `document.execCommand('copy')` → 클립보드.
- 자동화가 쉬웠다.

캔버스 에디터로 넘어가면서 이 모든 게 무용지물.

### 💡 어떻게 해결했는가 — 자동화 전략들

**(1) 에디터 내부 함수 호출**
- "전체 텍스트 가져오기" 함수를 에디터 자체에 노출하거나 후킹.

**(2) Document Model 직접 접근**
- 메모리에서 Model 객체를 찾아 읽기.
- → 이게 챌린지의 "breakthrough" 가능성.

**(3) 키 이벤트로 우회**
- `Ctrl+A` → 에디터의 select-all 트리거 → 에디터 내부 selection 사용.

**(4) 네트워크 가로채기**
- Save 시 서버로 보내는 페이로드 파싱.
- 단, 항상 가능한 건 아님 (실시간 협업 binary 등).

### 🔬 챌린지의 함의

> 표준 API에 의존하는 모든 솔루션은 **막다른 길**.
> **에디터 내부로 들어가야 한다.** 이 인식이 출발점.

---

# 제3부 — 리버스 엔지니어링 기법

## 9장. Chrome DevTools 핵심 패널들

### 🎯 한 줄 정의

DevTools(F12로 여는 개발자 도구)는 **브라우저 안의 모든 것을 들여다보는 만능 현미경**. Sources, Network, Performance, Memory 가 핵심.

### 🎈 쉬운 비유

> 과학 실험실의 도구들:
> - **Sources** = **현미경** (코드 한 줄씩 들여다봄).
> - **Network** = **녹음기** (브라우저가 외부와 주고받는 모든 통신).
> - **Performance** = **속도계** (어디서 시간이 잡아먹히는지).
> - **Memory** = **CT 스캔** (메모리 안의 객체 구조).
> - **Console** = **즉석 실험대** (코드를 쳐서 즉시 결과 확인).

### 🤔 왜 필요한가

리버스 엔지니어링 = **블랙박스를 들여다보기**.
- 코드를 봐야 함 → Sources.
- 통신을 봐야 함 → Network.
- 성능을 봐야 함 → Performance.
- 메모리를 봐야 함 → Memory.

### 😣 옛날엔 무엇이 불편했나

옛날엔 도구가 별로 없었다:
- Firebug (Firefox 확장, 2006~2017) 가 처음으로 강력했음.
- 그 전에는 `alert()` 디버깅 시대.

### 💡 어떻게 해결했는가

**Chrome DevTools (2008~)** 가 발전하면서:
- 모든 패널이 통합.
- CDP를 통해 외부에서도 같은 기능 사용 가능.
- 거의 모든 웹 리버스 엔지니어링이 DevTools 위에서 시작.

### 🔬 패널별 활용 — 이 챌린지에 적용

#### Sources 패널
- **번들 파일 목록** 살펴보기 (`webhwp` 관련).
- **Pretty-print** 버튼으로 minified 코드 가독화.
- **Search in all files** (Cmd+Opt+F): 키워드로 전체 검색.
  - 예: "paragraph", "table", "save", "model" 같은 키워드.
- **Breakpoint** 설정해서 함수 진입 시 변수 확인.

#### Network 패널
- 문서 로드 시 받는 요청 캡처.
- JSON 페이로드의 구조 분석.
- WebSocket 메시지 (협업용일 가능성).
- Save 시 보내는 페이로드.

#### Performance 패널
- 타이핑 한 글자 입력 시의 콜스택 캡처.
- 어떤 함수가 호출되는지.

#### Memory 패널
- **Heap Snapshot** 으로 메모리 안 객체 그래프 캡처.
- "Document" 같은 큰 객체 찾기.
- Retainer로 그 객체가 어디서 참조되는지 추적.

#### Console
- `window` 객체 탐색.
- 한 줄 코드로 즉석 실험.
- `monitor()`, `monitorEvents()` 같은 디버깅 함수.

---

## 10장. 흔한 진입점들 — 어디서부터 코드를 파고들 것인가

### 🎯 한 줄 정의

리버스 엔지니어링의 **출발점이 될 만한 곳들의 카탈로그.** 전역 변수, 소스맵, WebAssembly, 메시지 채널 등.

### 🎈 쉬운 비유

> 미로에 들어갈 때 **입구를 잘 고르는 게 절반**.
> 어떤 입구는 막다른 길로, 어떤 입구는 보물로 이어진다.

### 🤔 왜 필요한가

코드 베이스가 거대할 때 (한컴 닥스도 수십만 줄), **무작위 탐색은 시간 낭비**.
- 패턴을 알면 진입점을 빨리 찾음.

### 😣 옛날엔 무엇이 불편했나

처음 리버스 엔지니어링 하는 사람은:
- 아무 파일이나 열어보다 길을 잃음.
- 며칠씩 헤매는 일.

### 💡 어떻게 해결했는가 — 진입점 카탈로그

#### (1) 전역 변수 / 싱글톤
```javascript
// Console에서:
Object.keys(window).filter(k => !knownGlobals.includes(k))
```
- `window.app`, `window.editor`, `window.__HWP__` 같은 흔적.
- 개발자가 디버깅용으로 붙여놨을 가능성.

#### (2) 소스맵 (.map 파일)
- minified 코드 옆에 `//# sourceMappingURL=...` 주석.
- 있으면 **원본 소스 그대로 복원** 가능 (변수명, 함수명).
- DevTools가 자동으로 사용.

#### (3) WebAssembly 모듈
- 한컴은 HWP 엔진을 **wasm으로 컴파일** 했을 가능성 매우 높음.
- DevTools Sources에서 `.wasm` 파일 검색.
- `WebAssembly.Module` 사용 흔적 grep.
- wasm은 **export된 함수명이 단서** (예: `_get_paragraph_text`).

#### (4) PostMessage / BroadcastChannel
- iframe 간 통신 / 워커 통신.
- 메시지 캡처하면 내부 명령 구조 파악 가능.
```javascript
window.addEventListener('message', e => console.log(e));
```

#### (5) IndexedDB / LocalStorage
- 로컬에 캐시된 문서 데이터.
- DevTools Application 탭에서 확인.

#### (6) Service Worker / Web Worker
- 워커 안에 핵심 로직이 있을 수 있음.
- DevTools에서 워커 컨텍스트로 진입 가능.

### 🔬 우선순위 (이 챌린지)

1. **전역 변수** — 1분 만에 확인 가능.
2. **소스맵** — 있으면 행운, 없으면 next.
3. **WebAssembly 존재 여부** — 한컴은 가능성 매우 높음.
4. **postMessage** — 워커/iframe 통신.
5. **Network** — 초기 페이로드 분석.

---

## 11장. 함수 후킹 / 몽키패칭

### 🎯 한 줄 정의

**원본 함수를 우리가 만든 함수로 감싸서, 호출 인자/리턴값을 가로채거나 동작을 변경하는 기법.**

### 🎈 쉬운 비유

> 우체부가 매일 우편물을 배달한다.
> 누군가 **우체부 옆에 따라가면서 무슨 편지를 누구에게 배달하는지 적어둔다**.
> 우체부는 평소처럼 일하지만, 모든 행동이 기록된다.
>
> 이게 후킹.

### 🤔 왜 필요한가

리버스 엔지니어링에서:
- 어떤 함수가 호출되는지.
- 무엇을 인자로 받는지.
- 무엇을 리턴하는지.

이걸 알아야 동작을 이해할 수 있다.

### 😣 옛날엔 무엇이 불편했나

옛날엔 **소스코드를 직접 수정** 해야 했다:
- 빌드 다시.
- 배포.
- 시간 낭비.

### 💡 어떻게 해결했는가

**JavaScript는 일등급 함수** 라서 후킹이 매우 쉽다:

```javascript
// 원본 보관
const original = window.someObject.someFunction;

// 감싸기
window.someObject.someFunction = function(...args) {
  console.log('Called with:', args);
  const result = original.apply(this, args);
  console.log('Returned:', result);
  return result;
};
```

이 한 패턴으로 모든 함수의 호출을 추적할 수 있다.

### 🔬 이 챌린지의 활용 시나리오

#### 시나리오 A: "어떤 함수가 텍스트를 받는지" 찾기
1. 키 입력 핸들러를 후킹해서 인자 캡처.
2. 그 인자가 어디로 흘러가는지 추적.
3. Document Model 의 변경 함수 발견.

#### 시나리오 B: "Save 시 무엇을 보내는지" 추적
1. `fetch`, `XMLHttpRequest`, `WebSocket.send` 를 후킹.
2. Save 버튼 누른 후의 통신 캡처.
3. 페이로드 구조 분석.

#### 시나리오 C: "Document Model 변경 추적"
1. 의심되는 객체의 setter들을 후킹.
2. 변경마다 stack trace 출력.
3. 변경 진입점 파악.

### 🔥 강력한 패턴 — Proxy

```javascript
window.someObject = new Proxy(window.someObject, {
  get(target, prop) {
    console.log(`Reading ${prop}`);
    return target[prop];
  },
  set(target, prop, value) {
    console.log(`Setting ${prop} = `, value);
    target[prop] = value;
    return true;
  },
});
```

객체의 모든 접근을 가로챔.

---

## 12장. 소스맵 복원, Webpack Chunk 추적, Scope Chain

### 🎯 한 줄 정의

- **소스맵**: minified 코드를 원본으로 매핑하는 파일.
- **Webpack chunk**: 코드 분할 (code splitting) 의 결과물.
- **Scope chain**: JavaScript의 변수 탐색 경로 (closure, module scope).

### 🎈 쉬운 비유

> 외국어로 번역된 책에 **원어 페이지 번호 표시** 가 있는 게 소스맵.
> 책이 너무 길어서 **챕터별로 나눠 출판** 한 게 webpack chunk.
> 책 안에서 인용된 단어가 **앞 챕터의 어디에 있었는지** 찾는 게 scope chain.

### 🤔 왜 필요한가

현대 웹 앱은:
- **번들링 / minification** 으로 코드를 압축 (변수명이 a, b, c 같은 한 글자).
- **코드 분할** 로 여러 파일 (lazy loading).
- **모듈 시스템** 으로 변수 캡슐화.

읽으려면 이 구조를 풀어야 함.

### 😣 옛날엔 무엇이 불편했나

minified 코드:
```javascript
function a(b,c){return b.d.e(c)}
```
- 무슨 일을 하는지 전혀 모름.
- 변수가 a, b, c.

### 💡 어떻게 해결했는가

**(1) 소스맵 자동 적용**
- DevTools Sources가 .map 파일을 발견하면 자동으로 원본 표시.
- 변수명 복원, 라인 매핑.

**(2) Webpack chunk 분석**
- `webpack/runtime`, `chunkLoadingGlobal` 같은 흔적.
- 동적 import (`import()`) 가 chunk 로드를 트리거.
- DevTools의 Coverage 탭으로 어떤 chunk가 활성인지 확인.

**(3) Scope chain 탐색**
- 중단점에서 멈추면 DevTools가 **현재 스코프의 모든 변수** 표시.
- closure로 숨겨진 객체에 접근 가능.

### 🔬 한컴 닥스에 적용

- 소스맵 있는지 확인 (대부분 production은 없음, 가끔 노출됨).
- Webpack chunk 패턴인지 확인.
- 핵심 함수에 breakpoint → 그 시점의 scope chain에서 Document Model 발견 가능성.

---

# 제4부 — 워드프로세서 도큐먼트 모델 일반론

## 13장. HWP / HWPX 포맷의 큰 그림

### 🎯 한 줄 정의

**한국 한컴 워드프로세서의 문서 포맷.** HWP는 바이너리, HWPX는 OOXML 스타일의 XML+ZIP 포맷.

### 🎈 쉬운 비유

> 책의 구조와 같다:
> - **Section** (장) → **Paragraph** (단락) → **Run** (서식이 같은 글자 묶음) → **Character** (글자).
> - 옆에 **Style table** (글꼴, 색깔 등 사전).
> - 옆에 **Asset** (이미지, 표 등).

### 🤔 왜 알아야 하는가

자동화 SDK는 **포맷의 구조** 를 알아야 한다:
- "단락" 이 무엇인지.
- "서식" 을 어떻게 표현할지.
- 마크다운으로 변환할 때 어떻게 매핑할지.

### 😣 옛날엔 무엇이 불편했나

워드 프로세서마다 포맷이 다 달랐다:
- DOC (Word, 바이너리).
- HWP (한컴, 바이너리).
- ODT (LibreOffice, ZIP+XML).
- PDF (변환).
- 호환성 지옥.

### 💡 어떻게 해결했는가

**XML+ZIP 표준화 흐름**:
- DOCX (Word, 2007~).
- HWPX (한컴, 2010~).
- 둘 다 **XML 안에 단락/서식 정보** + ZIP으로 묶음.
- 파싱이 훨씬 쉬워짐.

### 🔬 일반적 구조

```
Document
 ├── Section
 │    ├── Paragraph
 │    │    ├── Run (font: 맑은고딕, size: 11pt, bold: true)
 │    │    │    └── "안녕하세요"
 │    │    └── Run (font: 맑은고딕, size: 11pt, bold: false)
 │    │         └── ", 반갑습니다."
 │    ├── Paragraph (style: Heading 1)
 │    │    └── Run
 │    │         └── "제목"
 │    ├── Table
 │    │    ├── Row
 │    │    │    ├── Cell (Paragraph...)
 │    │    │    └── Cell (Paragraph...)
 │    │    └── Row ...
 │    └── Image (src, width, height)
 └── Styles
      ├── "Heading 1": { font: ..., size: 18, bold: true }
      └── "Normal": { ... }
```

이 구조를 머릿속에 두면 **에디터 내부 객체** 도 비슷한 형태일 거라 예상 가능.

### 🔥 챌린지에서

- Document Model 객체를 찾으면 **거의 위 구조** 를 띌 가능성 높음.
- 그러면 read 기능들이 자연스럽게 매핑됨.

---

## 14장. 마크다운 매핑 — 워드 → MD 변환

### 🎯 한 줄 정의

**워드 프로세서의 풍부한 서식을 마크다운의 단순한 표현으로 변환하는 규칙.**

### 🎈 쉬운 비유

> **그림 일기 → 일기장 글** 으로 옮겨 적는 작업.
> 그림이 가진 색깔, 크기, 위치 정보를 → 글의 단순한 단어로 표현.
> 정보 손실은 불가피, 그러나 **본질은 보존**.

### 🤔 왜 필요한가

문서를 다른 시스템(블로그, 메모, GitHub)에 옮길 때 **공용어** 가 필요.
- 마크다운은 **사실상 표준**.

### 😣 옛날엔 무엇이 불편했나

문서 변환은 늘 **정보 손실 vs 호환성** 갈등.
- 너무 많이 보존하려 하면 비표준 syntax.
- 너무 적게 보존하면 본질 잃음.

### 💡 어떻게 해결했는가 — 매핑 규칙들

#### 헤딩
- 스타일 이름 기반 (`Heading 1` → `#`, `Heading 2` → `##`).
- 또는 폰트 크기 기반 (큰 글자 → 헤딩으로 추정).

#### Bold / Italic
- `<strong>`, `<em>` 으로 표현.
- 마크다운은 `**bold**`, `*italic*`.

#### 표
- HTML table → 마크다운 `| col | col |`.
- 단순한 표만 마크다운, 복잡한 건 HTML 유지.

#### 이미지
- `![alt](src)` 또는 base64 inline.

#### 색깔, 폰트, 크기
- **마크다운에 없음**. 잃거나, 확장 syntax 사용.

### 🔬 인접한 Run 합치기

원본 Document Model에서:
```
Run("안녕", bold)
Run("하세요", bold)
```
- 그대로 두면 마크다운에서 `**안녕****하세요**` (이상함).
- **인접한 같은 서식 Run을 먼저 합쳐야** → `**안녕하세요**`.

### 🔥 이 챌린지에서

- 마크다운 export는 **순수 변환 로직** → 단위 테스트 좋음.
- Document Model을 잘 추출했다면 매핑은 비교적 직관적.

---

## 15장. OT / CRDT — 협업 에디터의 동기화

### 🎯 한 줄 정의

- **OT (Operational Transformation)**: 동시 편집 시 충돌하는 편집을 변환해서 일관성 보장.
- **CRDT (Conflict-free Replicated Data Type)**: 충돌 자체가 일어나지 않는 자료구조.

### 🎈 쉬운 비유

> 두 사람이 같은 메뉴판을 동시에 수정한다.
> - 한 사람이 "치즈" 추가, 동시에 다른 사람이 "라떼" 추가.
> - **OT**: 두 변경을 받아서 적절히 병합 (서로의 위치를 고려해 조정).
> - **CRDT**: 처음부터 충돌이 안 나도록 자료구조를 설계 (예: 위치를 분수로 표현).

### 🤔 왜 필요한가

요즘 워드 프로세서는 거의 다 **실시간 협업** 지원.
- Google Docs, Notion, Hancom Docs.
- 동시 편집 시 일관성 필요.

### 😣 옛날엔 무엇이 불편했나

옛날엔 **lock 기반** 협업.
- 한 사람만 편집 가능.
- 다른 사람은 대기.
- 사용성 끔찍.

### 💡 어떻게 해결했는가

**(1) OT (1989, Ellis & Gibbs)**
- Google Docs가 사용.
- 모든 변경을 "operation" 으로 표현.
- 서버에서 transform 함수로 병합.

**(2) CRDT (2011~)**
- Yjs, Automerge 같은 라이브러리.
- Local-first 설계 가능.
- 분산 친화적.

### 🔬 자동화 관점에서의 함의

- 한컴 닥스가 협업이라면:
  - **편집은 "operation/command" 단위** 로 일어남.
  - 키 이벤트 → command 생성 → 서버로 전송 → 모든 클라이언트에 반영.
- 자동화 SDK는 이 command 흐름을 알면 더 정밀한 제어 가능.
- **Network 탭에서 WebSocket 메시지 분석** 시 op 형태가 보일 가능성.

### 🔥 이 챌린지에서

- 협업 메시지 형식을 알면 **그걸 직접 보내서** 텍스트 삽입도 가능 (키 이벤트 우회).
- 단, 인증/세션 처리 복잡.
- 일반적으로는 키 이벤트 방식이 더 robust.

---

# 제5부 — SDK 설계 측면

## 16장. 클라이언트 클래스 설계

### 🎯 한 줄 정의

**CDP의 저수준 명령들을 사용하기 좋은 고수준 메서드로 감싸는 클래스 설계.**

### 🎈 쉬운 비유

> 자동차 운전:
> - **저수준**: 엔진 RPM 조절, 변속 타이밍, 클러치 압력 …
> - **고수준**: "출발", "정지", "좌회전".
>
> 사용자는 고수준 명령만 알면 운전할 수 있다.
> SDK도 같다.

### 🤔 왜 필요한가

raw CDP를 그냥 노출하면:
- 사용자가 매번 도메인/메서드 이름 외워야 함.
- WebSocket 연결, 재연결 직접 처리.
- 에러 처리 일관성 없음.
- 테스트 어려움.

### 😣 옛날엔 무엇이 불편했나

라이브러리 없이 직접 짜면 매 프로젝트마다 boilerplate 반복.

### 💡 어떻게 해결했는가 — 좋은 클래스 설계 패턴

```typescript
class HancomDocsClient {
  // 연결 관리
  async connect(port: number): Promise<void>
  async disconnect(): Promise<void>

  // 고수준 read
  async getFullText(): Promise<string>
  async getStructure(): Promise<DocumentNode>
  async getFormatAt(paragraphIndex: number): Promise<TextFormat>
  async search(query: string): Promise<SearchResult[]>
  async exportMarkdown(): Promise<string>

  // 고수준 write
  async typeText(text: string): Promise<void>
  async findAndReplace(find: string, replace: string): Promise<void>
  async insertTable(rows: number, cols: number): Promise<void>
  async fillCell(text: string): Promise<void>
  async save(): Promise<void>

  // 저수준 (escape hatch)
  async evaluate<T>(expression: string): Promise<T>
  async sendCDP(method: string, params: any): Promise<any>
}
```

**원칙**:
- 사용자는 고수준 메서드만 보면 됨.
- 막히면 저수준 escape hatch 사용 가능.
- 에러는 통일된 Error 타입으로.

### 🔬 추가 디자인 고려

- **Connection lifecycle**: connect/disconnect, auto-reconnect.
- **Target selection**: 여러 탭 중 어떤 걸 고를지.
- **Event subscription**: 페이지 변화 알림.
- **Logging / debug mode**: 디버깅 시 모든 CDP 메시지 출력.

---

## 17장. 비동기 모델 — JSON-RPC와 ID 매칭

### 🎯 한 줄 정의

**CDP는 비동기 메시지 기반.** 요청과 응답이 ID로 매칭된다.

### 🎈 쉬운 비유

> 식당에서 주문:
> - 손님이 **번호표** 를 받음 (request id).
> - 음식이 나오면 번호표로 호출.
> - 여러 손님의 주문이 동시에 처리됨.
>
> 동기 방식이라면 한 사람씩 줄 서서 음식 받고 다음 사람.
> 비동기는 동시에 여러 일이 진행됨.

### 🤔 왜 필요한가

WebSocket은 **양방향, 비동기**:
- 여러 요청이 동시에 보낼 수 있음.
- 응답 순서가 요청 순서와 다를 수 있음.
- ID로 매칭해야 함.

### 😣 옛날엔 무엇이 불편했나

비동기 처리 안 하면:
- 첫 요청 응답 기다리는 동안 다른 요청 못 보냄.
- 매우 느림.

### 💡 어떻게 해결했는가

**Promise / async-await 패턴**:

```typescript
class CDPClient {
  private nextId = 1;
  private pending = new Map<number, (result: any) => void>();

  send(method: string, params: any): Promise<any> {
    const id = this.nextId++;
    return new Promise((resolve) => {
      this.pending.set(id, resolve);
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  onMessage(msg: string) {
    const data = JSON.parse(msg);
    if (data.id && this.pending.has(data.id)) {
      this.pending.get(data.id)!(data.result);
      this.pending.delete(data.id);
    } else if (data.method) {
      // 이벤트 처리
      this.emit(data.method, data.params);
    }
  }
}
```

- `nextId++` 로 고유 ID.
- Promise를 Map에 저장.
- 응답 오면 Promise resolve.

### 🔬 더 깊이

- **Timeout** 처리 (응답이 안 오면 reject).
- **Cancellation** (요청 취소).
- **Backpressure** (너무 많은 요청 보내지 않기).
- **Error 전파** (응답에 `error` 필드 있을 때).

---

## 18장. Wait 전략 — 캔버스 에디터에서 동기화

### 🎯 한 줄 정의

**키 입력 후 에디터가 처리할 시간을 기다리는 전략.** DOM이 안 바뀌니 다른 신호를 봐야 한다.

### 🎈 쉬운 비유

> 누군가에게 메시지를 보내고 답장을 기다린다.
> - **카톡**: "읽음" 표시가 뜸 → 동기화 신호.
> - **이메일**: 답장이 와야 → 신호 약함.
>
> 에디터도 같다. **신호** 가 있어야 다음 동작 안전.

### 🤔 왜 필요한가

자동화에서 흔한 실수:
- 키 입력 → **즉시 다음 동작** → 에디터가 미처 처리 못 해서 누락.

### 😣 옛날엔 무엇이 불편했나

옛날엔 **고정 sleep** 사용:
- "0.1초 기다리고 다음 키" → 너무 짧으면 누락, 너무 길면 느림.
- Race condition.

### 💡 어떻게 해결했는가 — 전략들

#### (1) Polling (가장 단순)
```javascript
async function waitFor(condition, timeout = 5000) {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    if (await condition()) return true;
    await sleep(50);
  }
  throw new Error('timeout');
}

// 사용
await client.typeText("a");
await waitFor(async () => {
  const text = await client.evaluate("editor.getText()");
  return text.endsWith("a");
});
```

#### (2) Event listening (있다면 최선)
- 에디터가 발생시키는 이벤트 후킹.
- 변경 이벤트 받으면 다음 동작.

#### (3) Animation frame 기반
```javascript
// 다음 frame까지 기다림 (렌더링 완료 보장)
await client.evaluate("new Promise(r => requestAnimationFrame(() => r()))");
```

#### (4) Mutation observer (DOM이 살짝 바뀐다면)
- 캔버스 외곽 (메뉴, 상태바) 변화 감지.

### 🔬 이 챌린지에서

- **Document Model 객체에 직접 polling** 이 가장 신뢰성 높음.
- Memory에서 객체 찾았다면, 그 객체의 상태가 변하는 걸 감시.

---

## 19장. 테스트 전략 — 단위 vs 통합

### 🎯 한 줄 정의

**순수 함수는 단위 테스트, CDP 의존부는 통합 테스트로 분리.**

### 🎈 쉬운 비유

> 자동차 부품 테스트:
> - **단위 테스트** = 부품 하나하나 검사 (브레이크 패드 마찰력 등).
> - **통합 테스트** = 차 한 대 조립해서 도로 주행.
>
> 단위는 빠르고 자주, 통합은 느리지만 진짜 동작 검증.

### 🤔 왜 필요한가

CDP에 의존하는 코드는:
- Chrome 인스턴스 필요.
- 한컴 닥스 페이지 필요.
- **느리고 환경 의존적**.

→ 모든 테스트를 통합으로 하면 CI에서 죽음.

### 😣 옛날엔 무엇이 불편했나

옛날엔 분리 안 하고 다 통합 → 느리고 flaky.

### 💡 어떻게 해결했는가

**분리 원칙**:

#### 단위 테스트 가능한 것
- Document Model JSON → 마크다운 변환.
- 검색 매칭 알고리즘.
- 서식 추론 로직.
- → **빠름, 결정적, CI 친화적**.

#### 통합 테스트 (선택)
- 실제 Chrome + 한컴 닥스로 end-to-end.
- Smoke test 정도만.

#### Mock CDP
- CDP 응답을 mock해서 클라이언트 클래스 테스트.

### 🔬 챌린지의 산출물에 적용

> **"순수 파싱·변환 로직에 단위 테스트"** 가 챌린지의 명시적 요구.
> 마크다운 export, JSON 변환 같은 부분에 단위 테스트.

---

# 제6부 — 운영·디버깅

## 20장. 로그·트레이스 — CDP 메시지 dump

### 🎯 한 줄 정의

**CDP의 모든 송수신 메시지를 기록해서 문제 발생 시 재현/분석할 수 있게 하는 것.**

### 🎈 쉬운 비유

> 비행기의 **블랙박스**.
> 평소엔 안 보지만, 사고 시 **무슨 일이 있었는지 정확히 복원** 가능.

### 🤔 왜 필요한가

자동화는 **타이밍 의존적** 이라 디버깅 어려움.
- "왜 어제는 됐는데 오늘 안 되지?"
- 로그 없으면 원인 추측.

### 😣 옛날엔 무엇이 불편했나

`console.log` 만 뿌리다가:
- 너무 많아서 못 봄.
- 또는 너무 적어서 단서 없음.

### 💡 어떻게 해결했는가

**구조화된 로깅**:
```typescript
class CDPClient {
  send(method: string, params: any) {
    this.logger.debug({ direction: 'out', method, params });
    // ...
  }
  onMessage(msg: string) {
    this.logger.debug({ direction: 'in', raw: msg });
    // ...
  }
}
```

- **레벨**: trace / debug / info / warn / error.
- **JSON 형식** 으로 기계적 분석 가능.
- **DEBUG 환경변수**로 켜고 끄기.

### 🔬 추가

- **세션 ID** 같이 기록 → 여러 세션 분리.
- **타임스탬프** (ms 단위) → 타이밍 분석.
- **Replay 가능성** — 로그를 보고 재현 가능하게.

---

## 21장. 에러 모드 — 페이지 새로고침, 토큰 만료, Race Condition

### 🎯 한 줄 정의

**자동화 시스템이 만나는 흔한 실패 시나리오들.** 미리 알고 대비해야 robust한 SDK.

### 🎈 쉬운 비유

> 자동차의 **고장 모드**:
> - 타이어 펑크.
> - 연료 부족.
> - 브레이크 고장.
>
> 미리 대비하면 사고 안 남.

### 🤔 왜 알아야 하나

production에서 SDK가 죽으면:
- 사용자 데이터 손실 위험.
- 신뢰 붕괴.
- 디버깅 비용 폭증.

### 😣 옛날엔 무엇이 불편했나

happy path만 짜다가:
- 실제 환경에서 다양한 실패 → 매번 hot fix.

### 💡 어떻게 해결했는가 — 에러 카탈로그

#### (1) 페이지 새로고침
- WebSocket 연결 끊김.
- Document Model 객체 사라짐 (다시 찾아야 함).
- **대응**: 재연결 + 재초기화 로직.

#### (2) 토큰 만료
- 한컴 닥스 세션 만료.
- 명령은 가지만 **저장 실패**.
- **대응**: 저장 후 응답 검증.

#### (3) Race condition
- 여러 명령이 동시에 → 순서 꼬임.
- **대응**: 명령 큐, 직렬화.

#### (4) 사용자가 동시에 조작
- SDK가 타이핑하는 동안 사용자가 클릭.
- **대응**: 명령 시작 전 상태 확인, 가능하면 lock.

#### (5) 네트워크 지연
- 협업 동기화 지연.
- 명령 전송 → 즉시 다른 명령 → 충돌.
- **대응**: 적절한 wait.

#### (6) Chrome 크래시
- WebSocket 끊김.
- **대응**: graceful shutdown + error report.

### 🔬 robust SDK의 일반 원칙

- **Idempotent** 명령 우선 (같은 명령 두 번 = 한 번).
- **재시도 + backoff** (지수 백오프).
- **상태 검증 후 동작** (precondition check).
- **명확한 에러 메시지** (사용자가 디버깅 가능).

---

# 맺음말

이 책에서 다룬 21개 개념은 **Q2를 풀기 위한 최소한의 어휘 사전** 이다.
Q1과 마찬가지로, **개념 자체보다 더 중요한 건** 각 개념이 등장한 **맥락** 이다.

> "왜 캔버스 에디터로 갔나?" → "큰 문서에서 DOM이 너무 느려서."
> "왜 raw CDP를 쓰나?" → "Puppeteer가 캔버스의 미묘한 동작을 못 다뤄서."
> "왜 함수 후킹을 쓰나?" → "코드를 다시 빌드 안 하고 동작을 바꿀 수 있어서."

Q2는 Q1과 다른 결의 시험이다.
- Q1은 **수학적 / 시스템적 분석** 의 깊이.
- Q2는 **탐험과 직관** 의 깊이.

> "이 시스템 안에 **반드시 있을 법한 객체** 는 무엇인가?"
> "그것을 **어떻게 손에 넣을 것인가?**"
> "그것이 손에 들어오면 **모든 read 기능이 자연스럽게 풀리는 지점** 은 어디인가?"

이 세 질문이 Q2의 핵심.

이 책은 그 질문들을 던지기 위한 **사전 정비**다.
이제 Chrome을 띄워라.

---

## 참고 자료

- [`q2_ready.md`](./q2_ready.md) — 풀이 진입 전 체크리스트.
- [`task-web.ko.md`](./task-web.ko.md) — 챌린지 원문 번역.
- [`developer_guide.ko.md`](./developer_guide.ko.md) — 이 일을 하는 개발자가 누구인지.

### 추천 외부 자료

- Chrome DevTools Protocol 공식 문서: <https://chromedevtools.github.io/devtools-protocol/>
- Chrome DevTools Architecture: <https://developer.chrome.com/docs/devtools/>
- MDN — JavaScript Runtime: <https://developer.mozilla.org/en-US/docs/Web/JavaScript>
- "How Google Docs went from DOM to canvas": (구글 검색 — 2021년 글들)
- Chrome DevTools MCP: <https://developer.chrome.com/blog/chrome-devtools-mcp-debug-your-browser-session>
