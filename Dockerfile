FROM python:3.11-slim

WORKDIR /app
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHONHASHSEED=42 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml requirements.txt README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts

RUN python -m pip install --no-cache-dir . \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/outputs \
    && chown -R appuser:appuser /app

USER appuser
CMD ["support-capacity", "run", "--config", "configs/smoke.yaml", "--require-release"]
