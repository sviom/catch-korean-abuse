FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=0

# ffmpeg 설치 (Whisper 오디오 디코딩 필수)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock* /app/

RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS runtime

WORKDIR /app

# .venv만 복사해 런타임을 가볍게
COPY --from=installer /app/.venv /app/.venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

COPY . /app

RUN rm -rf /root/.cache ~/.cache /tmp/*

RUN python -c "import sys; print(sys.executable); import gradio; print('gradio ok')"

ENV PORT=7860
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["python", "app.py"]