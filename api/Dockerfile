FROM python:3.10

# 1. Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 2. Copy config files from Root
COPY pyproject.toml uv.lock ./

# 3. Install dependencies (uv handles the CPU/Linux logic automatically)
RUN uv sync --frozen --no-cache

# 4. Add venv to PATH (Crucial for uvicorn)
ENV PATH="/app/.venv/bin:$PATH"

# 5. Copy the 'api' folder structure
COPY api/ api/

# 6. Environment
ENV PYTHONPATH=/app
ENV ARTIFACT_PATH="/app/api/models_local"

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]