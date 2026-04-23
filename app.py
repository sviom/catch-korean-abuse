import gradio as gr
import numpy as np
import anthropic
import time
import json
import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path

# ── Whisper 로드 ──────────────────────────────────────────────────
import warnings

warnings.filterwarnings("ignore")

try:
    import whisper as openai_whisper

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("[WARN] openai-whisper not installed. Using mock transcription.")

# ── 설정 ─────────────────────────────────────────────────────────
DEFAULT_MODEL_TYPE = "base"  # "base" | "local"
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "./local_model")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── 한국어 욕설 블랙리스트 (1차 필터) ────────────────────────────
KO_BLACKLIST = [
    "씨발",
    "시발",
    "ㅅㅂ",
    "씹",
    "개새끼",
    "새끼",
    "ㅅㄲ",
    "병신",
    "ㅂㅅ",
    "미친",
    "ㅁㅊ",
    "지랄",
    "꺼져",
    "닥쳐",
    "죽어",
    "존나",
    "ㅈㄴ",
    "창녀",
    "보지",
    "자지",
    "년",
    "놈",
    "개년",
    "개놈",
    "썅",
    "빌어먹",
    "엿먹",
    "니애미",
    "니에미",
    "니미",
    "ㄴㅁ",
    "좆",
    "ㅈ같",
    "fuck",
    "shit",
    "bitch",
]

# ── Whisper 모델 싱글턴 ───────────────────────────────────────────
_whisper_cache: dict = {}


def get_whisper_model(model_type: str, local_path: str = ""):
    key = f"{model_type}:{local_path}"
    if key not in _whisper_cache:
        if not WHISPER_AVAILABLE:
            _whisper_cache[key] = None
            return None
        if model_type == "local" and local_path and Path(local_path).exists():
            print(f"[INFO] 로컬 모델 로드: {local_path}")
            _whisper_cache[key] = openai_whisper.load_model(local_path)
        else:
            print("[INFO] Whisper base 모델 로드 중...")
            _whisper_cache[key] = openai_whisper.load_model("base")
    return _whisper_cache[key]


# ── 전사 ──────────────────────────────────────────────────────────
def transcribe_audio(audio_path: str, model_type: str, local_path: str) -> str:
    model = get_whisper_model(model_type, local_path)
    if model is None:
        return "[Mock] 안녕하세요 테스트 문장입니다."
    result = model.transcribe(audio_path, language="ko", fp16=False)
    return result.get("text", "").strip()


# ── 블랙리스트 1차 필터 ───────────────────────────────────────────
def blacklist_check(text: str) -> tuple[bool, list[str]]:
    text_lower = text.lower()
    found = [w for w in KO_BLACKLIST if w in text_lower]
    return bool(found), found


# ── LLM 2차 판별 ─────────────────────────────────────────────────
def llm_check(text: str) -> tuple[bool, str, float]:
    """Claude API로 문맥 기반 욕설 판별. (is_profane, reason, confidence)"""
    if not ANTHROPIC_API_KEY:
        return False, "API 키 없음 - LLM 판별 건너뜀", 0.0

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = f"""다음 한국어 텍스트가 욕설/비속어/혐오 표현을 포함하는지 판별하세요.
초성체(ㅅㅂ, ㅂㅅ 등), 변형어, 우회 표현도 포함합니다.

텍스트: "{text}"

아래 JSON 형식으로만 응답하세요 (추가 설명 없이):
{{"is_profane": true/false, "confidence": 0.0~1.0, "reason": "간단한 이유"}}"""

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        data = json.loads(raw)
        return (
            bool(data["is_profane"]),
            data.get("reason", ""),
            float(data.get("confidence", 0.5)),
        )
    except Exception as e:
        return False, f"LLM 오류: {e}", 0.0


# ── 통합 판별 ─────────────────────────────────────────────────────
def detect_profanity(text: str, use_llm: bool) -> dict:
    if not text.strip():
        return {
            "is_profane": False,
            "method": "-",
            "reason": "빈 텍스트",
            "confidence": 0.0,
            "found_words": [],
        }

    # 1차
    bl_hit, found_words = blacklist_check(text)

    if bl_hit:
        return {
            "is_profane": True,
            "method": "블랙리스트",
            "reason": f"금지어 감지: {', '.join(found_words)}",
            "confidence": 1.0,
            "found_words": found_words,
        }

    # 2차 (선택)
    if use_llm:
        is_p, reason, conf = llm_check(text)
        return {
            "is_profane": is_p,
            "method": "LLM (Claude)",
            "reason": reason,
            "confidence": conf,
            "found_words": [],
        }

    return {
        "is_profane": False,
        "method": "블랙리스트만",
        "reason": "욕설 없음",
        "confidence": 0.95,
        "found_words": [],
    }


# ── 로그 포맷 ─────────────────────────────────────────────────────
def format_log_row(ts: str, text: str, result: dict) -> str:
    icon = "🚨" if result["is_profane"] else "✅"
    label = "욕설 감지" if result["is_profane"] else "정상"
    conf = f"{result['confidence']*100:.0f}%"
    method = result["method"]
    reason = result["reason"][:40]
    return (
        f"<tr class='{'row-profane' if result['is_profane'] else 'row-clean'}'>"
        f"<td>{ts}</td><td>{icon} {label}</td>"
        f"<td class='text-cell'>{text[:60]}{'…' if len(text)>60 else ''}</td>"
        f"<td>{conf}</td><td>{method}</td><td>{reason}</td></tr>"
    )


# ── 파일 업로드 처리 ──────────────────────────────────────────────
def process_uploaded_file(
    audio_file,
    interval_sec: float,
    use_llm: bool,
    model_type: str,
    local_model_path: str,
    log_state: list,
    stats_state: dict,
):
    if audio_file is None:
        return (
            log_state,
            stats_state,
            build_log_html(log_state),
            build_stats_html(stats_state),
            "파일을 업로드해 주세요.",
        )

    status_msgs = []
    try:
        import soundfile as sf

        data, sr = sf.read(audio_file)
        if data.ndim > 1:
            data = data.mean(axis=1)

        total_samples = len(data)
        chunk_samples = int(interval_sec * sr)
        num_chunks = max(1, total_samples // chunk_samples)

        status_msgs.append(
            f"📂 파일 로드 완료 | 총 {total_samples/sr:.1f}초 | {num_chunks}개 청크 처리 예정"
        )
        yield log_state, stats_state, build_log_html(log_state), build_stats_html(
            stats_state
        ), "\n".join(status_msgs)

        for i in range(num_chunks):
            chunk = data[i * chunk_samples : (i + 1) * chunk_samples]
            if len(chunk) < sr * 0.3:
                continue

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                import soundfile as sf2

                sf2.write(f.name, chunk, sr)
                tmp_path = f.name

            ts = datetime.now().strftime("%H:%M:%S")
            text = transcribe_audio(tmp_path, model_type, local_model_path)
            result = detect_profanity(text, use_llm)
            os.unlink(tmp_path)

            log_state.append({"ts": ts, "text": text, "result": result})
            stats_state["total"] += 1
            if result["is_profane"]:
                stats_state["profane"] += 1

            status_msgs.append(
                f"[{ts}] 청크 {i+1}/{num_chunks} | "
                f"{'🚨 욕설' if result['is_profane'] else '✅ 정상'} | {text[:30]}…"
            )
            yield log_state, stats_state, build_log_html(log_state), build_stats_html(
                stats_state
            ), "\n".join(status_msgs[-8:])

        status_msgs.append("✅ 파일 처리 완료")
        yield log_state, stats_state, build_log_html(log_state), build_stats_html(
            stats_state
        ), "\n".join(status_msgs[-8:])

    except Exception as e:
        yield log_state, stats_state, build_log_html(log_state), build_stats_html(
            stats_state
        ), f"❌ 오류: {e}"


# ── 마이크 스트리밍 처리 ──────────────────────────────────────────
def process_stream_chunk(
    audio_chunk,
    interval_sec: float,
    use_llm: bool,
    model_type: str,
    local_model_path: str,
    log_state: list,
    stats_state: dict,
    buffer_state: dict,
):
    if audio_chunk is None:
        return (
            log_state,
            stats_state,
            buffer_state,
            build_log_html(log_state),
            build_stats_html(stats_state),
            "",
        )

    sr, data = audio_chunk
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if data.max() > 1.0:
        data = data / 32768.0

    buffer_state["samples"].extend(data.tolist())
    buffer_state["sr"] = sr

    required = int(interval_sec * sr)
    status = ""

    if len(buffer_state["samples"]) >= required:
        chunk = np.array(buffer_state["samples"][:required], dtype=np.float32)
        buffer_state["samples"] = buffer_state["samples"][required:]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            import soundfile as sf

            sf.write(f.name, chunk, sr)
            tmp_path = f.name

        ts = datetime.now().strftime("%H:%M:%S")
        text = transcribe_audio(tmp_path, model_type, local_model_path)
        result = detect_profanity(text, use_llm)
        os.unlink(tmp_path)

        log_state.append({"ts": ts, "text": text, "result": result})
        stats_state["total"] += 1
        if result["is_profane"]:
            stats_state["profane"] += 1

        icon = "🚨 욕설 감지!" if result["is_profane"] else "✅ 정상"
        status = f"[{ts}] {icon} | {text[:40]}"

    return (
        log_state,
        stats_state,
        buffer_state,
        build_log_html(log_state),
        build_stats_html(stats_state),
        status,
    )


# ── HTML 빌더 ─────────────────────────────────────────────────────
def build_log_html(log_state: list) -> str:
    if not log_state:
        return "<p class='empty-msg'>아직 분석 결과가 없습니다.</p>"
    rows = "".join(
        format_log_row(e["ts"], e["text"], e["result"])
        for e in reversed(log_state[-50:])
    )
    return f"""
<div class='log-wrap'>
<table class='log-table'>
<thead><tr>
  <th>시간</th><th>판정</th><th>전사 텍스트</th><th>신뢰도</th><th>방법</th><th>사유</th>
</tr></thead>
<tbody>{rows}</tbody>
</table></div>"""


def build_stats_html(stats: dict) -> str:
    total = stats.get("total", 0)
    profane = stats.get("profane", 0)
    clean = total - profane
    rate = (profane / total * 100) if total else 0
    bar_w = f"{rate:.0f}%"
    return f"""
<div class='stats-grid'>
  <div class='stat-card total'><div class='stat-num'>{total}</div><div class='stat-label'>전체 청크</div></div>
  <div class='stat-card danger'><div class='stat-num'>{profane}</div><div class='stat-label'>욕설 감지</div></div>
  <div class='stat-card safe'><div class='stat-num'>{clean}</div><div class='stat-label'>정상</div></div>
  <div class='stat-card rate'>
    <div class='stat-num'>{rate:.1f}%</div><div class='stat-label'>욕설 비율</div>
    <div class='bar-bg'><div class='bar-fill' style='width:{bar_w}'></div></div>
  </div>
</div>"""


def clear_log(log_state, stats_state):
    log_state.clear()
    stats_state.update({"total": 0, "profane": 0})
    return (
        log_state,
        stats_state,
        build_log_html(log_state),
        build_stats_html(stats_state),
        "",
    )


# ── CSS ───────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --bg:        #0d0f14;
  --surface:   #141720;
  --surface2:  #1c2030;
  --border:    #2a2f40;
  --accent:    #4f8ef7;
  --danger:    #f74f6a;
  --safe:      #4fffa3;
  --warn:      #f7c44f;
  --text:      #e8ecf5;
  --muted:     #6b7491;
  --radius:    10px;
}

body, .gradio-container { background: var(--bg) !important; color: var(--text) !important; font-family: 'Noto Sans KR', sans-serif !important; }

/* Header */
.app-header { text-align:center; padding: 2rem 0 1rem; }
.app-title { font-size: 2rem; font-weight: 700; letter-spacing: -0.5px;
  background: linear-gradient(135deg, var(--accent), var(--safe));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.app-sub { color: var(--muted); font-size: 0.9rem; margin-top: 0.3rem; }

/* Tabs */
.tab-nav button { background: var(--surface2) !important; color: var(--muted) !important;
  border: 1px solid var(--border) !important; border-radius: var(--radius) !important;
  font-family: 'Noto Sans KR', sans-serif !important; }
.tab-nav button.selected { background: var(--accent) !important; color: #fff !important; }

/* Labels & Sliders */
label span, .label-wrap span { color: var(--text) !important; font-size: 0.85rem !important; }
input[type=range] { accent-color: var(--accent); }
input[type=number], textarea { background: var(--surface2) !important; border: 1px solid var(--border) !important;
  color: var(--text) !important; border-radius: var(--radius) !important; }

/* Buttons */
button.primary { background: var(--accent) !important; color: #fff !important;
  border: none !important; border-radius: var(--radius) !important;
  font-family: 'Noto Sans KR', sans-serif !important; font-weight: 600 !important; }
button.secondary { background: var(--surface2) !important; color: var(--muted) !important;
  border: 1px solid var(--border) !important; border-radius: var(--radius) !important; }
button.stop { background: var(--danger) !important; color: #fff !important;
  border: none !important; border-radius: var(--radius) !important; }

/* Status box */
.status-box textarea { background: var(--surface) !important; border: 1px solid var(--border) !important;
  color: var(--safe) !important; font-family: 'JetBrains Mono', monospace !important; font-size: 0.82rem !important; }

/* Stats */
.stats-grid { display:grid; grid-template-columns: repeat(4,1fr); gap:12px; padding:4px; }
.stat-card { background: var(--surface2); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 16px 12px; text-align:center; }
.stat-card.danger { border-color: var(--danger); }
.stat-card.safe   { border-color: var(--safe);   }
.stat-card.total  { border-color: var(--accent);  }
.stat-card.rate   { border-color: var(--warn);    }
.stat-num   { font-size: 2rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.stat-label { font-size: 0.75rem; color: var(--muted); margin-top:4px; }
.bar-bg  { background: var(--border); border-radius:4px; height:4px; margin-top:8px; overflow:hidden; }
.bar-fill{ background: var(--danger); height:100%; transition: width 0.4s ease; border-radius:4px; }

/* Log table */
.log-wrap { overflow-x:auto; max-height:420px; overflow-y:auto; }
.log-table { width:100%; border-collapse:collapse; font-size:0.82rem; }
.log-table thead th { background: var(--surface); color: var(--muted);
  padding:8px 12px; text-align:left; position:sticky; top:0; border-bottom:1px solid var(--border); }
.log-table tbody td { padding:7px 12px; border-bottom:1px solid var(--border); vertical-align:middle; }
.row-profane { background: rgba(247,79,106,0.08); }
.row-clean   { background: transparent; }
.row-profane:hover { background: rgba(247,79,106,0.15); }
.row-clean:hover   { background: rgba(79,142,247,0.07);  }
.text-cell { font-family: 'JetBrains Mono', monospace; font-size:0.78rem; max-width:220px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.empty-msg { color: var(--muted); text-align:center; padding:40px; font-size:0.9rem; }

/* Checkboxes & Radios */
.gr-checkbox, input[type=checkbox] { accent-color: var(--accent); }

/* Section labels */
.section-label { color: var(--muted); font-size: 0.78rem; text-transform: uppercase;
  letter-spacing: 1px; padding: 8px 0 4px; border-top: 1px solid var(--border); margin-top: 8px; }
"""

# ── Gradio UI ─────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="한국어 욕설 탐지기") as demo:

    # ── 상태 ──
    log_state = gr.State([])
    stats_state = gr.State({"total": 0, "profane": 0})
    buffer_state = gr.State({"samples": [], "sr": 16000})

    # ── 헤더 ──
    gr.HTML("""
    <div class='app-header'>
      <div class='app-title'>🔍 한국어 욕설 탐지기</div>
      <div class='app-sub'>Whisper 기반 음성 전사 + 블랙리스트/LLM 2단계 판별 시스템</div>
    </div>
    """)

    with gr.Row():
        # ── 왼쪽: 설정 패널 ──────────────────────────────────────
        with gr.Column(scale=1, min_width=260):
            gr.HTML("<div class='section-label'>⚙️ 모델 설정</div>")
            model_type = gr.Radio(
                choices=["base", "local"],
                value="base",
                label="Whisper 모델",
                info="base = openai/whisper-base | local = 로컬 파인튜닝 모델",
            )
            local_path = gr.Textbox(
                label="로컬 모델 경로",
                placeholder="./local_model  (model_type=local 시 사용)",
                value=LOCAL_MODEL_PATH,
                visible=True,
            )

            gr.HTML("<div class='section-label'>🔎 판별 설정</div>")
            use_llm = gr.Checkbox(
                label="LLM 2차 판별 활성화 (Claude API)",
                value=bool(ANTHROPIC_API_KEY),
                info="블랙리스트 미감지 시 Claude API로 문맥 판별",
            )
            api_key_box = gr.Textbox(
                label="Anthropic API Key",
                placeholder="sk-ant-…  (환경변수 ANTHROPIC_API_KEY로도 설정 가능)",
                value=ANTHROPIC_API_KEY,
                type="password",
            )

            gr.HTML("<div class='section-label'>⏱️ 청크 설정</div>")
            interval = gr.Slider(
                minimum=1,
                maximum=30,
                value=5,
                step=1,
                label="분석 주기 (초)",
                info="오디오를 이 간격으로 잘라 분석합니다",
            )

            gr.HTML("<div class='section-label'>🗑️ 초기화</div>")
            clear_btn = gr.Button("로그 초기화", variant="secondary", size="sm")

        # ── 오른쪽: 입력 + 결과 ──────────────────────────────────
        with gr.Column(scale=3):
            with gr.Tabs():
                # ── 파일 업로드 탭 ────────────────────────────────
                with gr.Tab("📁 파일 업로드"):
                    file_audio = gr.Audio(
                        label="오디오 파일 업로드 (WAV / MP3 / M4A 등)",
                        type="filepath",
                        sources=["upload"],
                    )
                    analyze_btn = gr.Button("🔍 분석 시작", variant="primary")
                    file_status = gr.Textbox(
                        label="처리 상태",
                        lines=6,
                        interactive=False,
                        elem_classes=["status-box"],
                    )

                # ── 실시간 마이크 탭 ──────────────────────────────
                with gr.Tab("🎙️ 실시간 마이크"):
                    mic_audio = gr.Audio(
                        label="마이크 입력 (스트리밍)",
                        sources=["microphone"],
                        streaming=True,
                    )
                    mic_status = gr.Textbox(
                        label="실시간 상태",
                        lines=3,
                        interactive=False,
                        elem_classes=["status-box"],
                    )

            # ── 통계 ──────────────────────────────────────────────
            gr.HTML("<div class='section-label'>📊 통계</div>")
            stats_html = gr.HTML(build_stats_html({"total": 0, "profane": 0}))

            # ── 로그 ──────────────────────────────────────────────
            gr.HTML("<div class='section-label'>📋 분석 로그</div>")
            log_html = gr.HTML("<p class='empty-msg'>아직 분석 결과가 없습니다.</p>")

    # ── API Key 즉시 반영 ─────────────────────────────────────────
    def update_api_key(key):
        global ANTHROPIC_API_KEY
        ANTHROPIC_API_KEY = key.strip()
        return gr.update(value=bool(key.strip()))

    api_key_box.change(update_api_key, inputs=[api_key_box], outputs=[use_llm])

    # ── 모델 타입 변경 ────────────────────────────────────────────
    model_type.change(
        lambda t: gr.update(visible=(t == "local")),
        inputs=[model_type],
        outputs=[local_path],
    )

    # ── 파일 분석 ─────────────────────────────────────────────────
    analyze_btn.click(
        process_uploaded_file,
        inputs=[
            file_audio,
            interval,
            use_llm,
            model_type,
            local_path,
            log_state,
            stats_state,
        ],
        outputs=[log_state, stats_state, log_html, stats_html, file_status],
    )

    # ── 마이크 스트리밍 ───────────────────────────────────────────
    mic_audio.stream(
        process_stream_chunk,
        inputs=[
            mic_audio,
            interval,
            use_llm,
            model_type,
            local_path,
            log_state,
            stats_state,
            buffer_state,
        ],
        outputs=[
            log_state,
            stats_state,
            buffer_state,
            log_html,
            stats_html,
            mic_status,
        ],
    )

    # ── 초기화 ────────────────────────────────────────────────────
    clear_btn.click(
        clear_log,
        inputs=[log_state, stats_state],
        outputs=[log_state, stats_state, log_html, stats_html, file_status],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False,
    )
