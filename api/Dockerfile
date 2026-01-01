# 1. Base Image
FROM python:3.10-slim-bullseye

# 2. Set Working Directory
WORKDIR /app

# 3. Dependencies using pyproject
# We copy the file from the root context (since docker build context is .)
COPY pyproject.toml .

# Install dependencies directly from pyproject.toml
# This ensures Docker has the same libraries as GitHub Actions
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# 4. Copy API Code
COPY api/ api/

# 5. Copy Artifacts ('download-artifact' action)
COPY api/models_local/ /app/models_local/

# 6. Set Environment Variable
ENV ARTIFACT_PATH="/app/models_local"

# 7. Expose Port 8000 (Standard for Render/FastAPI)
EXPOSE 8000

# 8. Run Command
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"