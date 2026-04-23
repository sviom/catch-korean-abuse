# 🔍 한국어 욕설 탐지기

Whisper 기반 음성 전사 + 2단계 욕설 판별 시스템

---

## 아키텍처

```
오디오 입력 (파일 / 마이크 스트리밍)
       │
       ▼
  청크 분할 (사용자 설정 주기: 1~30초)
       │
       ▼
  Whisper 전사 (base 또는 로컬 파인튜닝 모델)
       │
       ▼
  ┌─ 1단계: 블랙리스트 필터 (즉시) ─────────────────────┐
  │  한국어 욕설 사전 + 초성체 + 변형어 매칭             │
  │  감지 시 → 즉시 결과 반환 (신뢰도 100%)             │
  └──────────────────────────────────────────────────────┘
       │ 미감지 시
       ▼
  ┌─ 2단계: LLM 문맥 판별 (선택) ───────────────────────┐
  │  Claude API 호출                                      │
  │  - 문맥 기반 우회 표현 탐지                          │
  │  - 신뢰도 점수 반환                                  │
  └──────────────────────────────────────────────────────┘
       │
       ▼
  결과 로그 (시간, 전사 텍스트, 판정, 신뢰도, 방법)
```

---

## 설치 및 실행

### 로컬 실행

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. ffmpeg 설치 (OS별)
# Ubuntu: sudo apt install ffmpeg
# macOS:  brew install ffmpeg
# Windows: https://ffmpeg.org/download.html

# 3. 환경변수 설정 (선택: LLM 판별 사용 시)
export ANTHROPIC_API_KEY="sk-ant-..."

# 4. 실행
python app.py
# → http://localhost:7860 접속
```

### Docker 실행

```bash
docker build -t profanity-detector .
docker run -p 7860:7860 \
  -e ANTHROPIC_API_KEY="sk-ant-..." \
  profanity-detector
```

---

## 클라우드 배포

### GCP Cloud Run

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/profanity-detector
gcloud run deploy profanity-detector \
  --image gcr.io/PROJECT_ID/profanity-detector \
  --platform managed \
  --port 7860 \
  --memory 4Gi \
  --set-env-vars ANTHROPIC_API_KEY="sk-ant-..."
```

### AWS App Runner / ECS

- 메모리 최소 4GB 권장 (Whisper base 모델 로드)
- GPU 없이도 동작하나 CPU는 약 2~5초/청크 소요

### HuggingFace Spaces

```bash
# Space 생성 후 파일 업로드
# README.md 상단에 아래 추가:
---
title: Korean Profanity Detector
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
---
```

---

## 로컬 파인튜닝 모델 사용

1. Whisper base를 한국어 욕설 데이터로 파인튜닝
2. 모델 저장 경로 지정 (예: `./local_model`)
3. UI에서 모델 타입을 `local`로 변경 후 경로 입력

파인튜닝 예시 (Hugging Face Trainer):
```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainer

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
# ... 데이터셋 준비 및 학습
model.save_pretrained("./local_model")
```

---

## 설정 옵션

| 항목 | 기본값 | 설명 |
|------|--------|------|
| 분석 주기 | 5초 | 오디오를 이 간격으로 잘라 분석 |
| Whisper 모델 | base | base 또는 로컬 경로 |
| LLM 판별 | 활성화 (API 키 있을 때) | 블랙리스트 미감지 시 Claude로 2차 판별 |

---

## 욕설 판별 방식

| 방식 | 속도 | 정확도 | 비용 |
|------|------|--------|------|
| 블랙리스트 1차 | 즉시 | 명시적 욕설만 | 무료 |
| Claude LLM 2차 | ~1초 | 우회/문맥 표현 포함 | API 비용 |
