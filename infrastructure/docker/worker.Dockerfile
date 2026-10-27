FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md LICENSE alembic.ini ./
COPY rag ./rag
COPY apps/api ./apps/api
# CPU-only torch first; otherwise pip resolves the CUDA build (several GB).
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install . \
    && groupadd --gid 10001 kip \
    && useradd --uid 10001 --gid kip --no-create-home kip \
    && mkdir -p /app/data \
    && chown kip:kip /app/data
USER kip
CMD ["kip-worker"]
