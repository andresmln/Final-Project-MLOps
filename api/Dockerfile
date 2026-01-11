# 1. Base Image
FROM python:3.10-slim-bullseye

# 2. Set Working Directory
WORKDIR /app

# 3. Install Dependencies FIRST (The Cache Fix)
COPY requirements.txt . 

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. Copy Code & Configuration
# This copies everything (including pyproject.toml, api/ folder, etc.)
COPY . api/

# 5. Set Environment Variable
ENV PYTHONPATH=/app
ENV ARTIFACT_PATH="/app/api/models_local"

# 6. Expose Port
EXPOSE 8000

# 7. Run Command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]