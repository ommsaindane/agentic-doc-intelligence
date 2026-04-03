# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create a virtual environment for deterministic runtime deps.
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install uv (for lockfile-based export) and upgrade pip.
RUN pip install --no-cache-dir -U pip uv

WORKDIR /app

# Copy only dependency manifests first for better layer caching.
COPY pyproject.toml uv.lock ./

# Export locked production requirements and install.
RUN uv export --frozen --no-dev -o requirements.txt \
 && pip install --no-cache-dir -r requirements.txt

# Copy application code.
COPY . /app


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Non-root user.
RUN useradd --create-home --shell /usr/sbin/nologin appuser

ENV VIRTUAL_ENV=/opt/venv
COPY --from=builder $VIRTUAL_ENV $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app
COPY --from=builder /app /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
