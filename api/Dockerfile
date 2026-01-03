# 1. Base Image
FROM python:3.10-slim-bullseye

# 2. Set Working Directory
WORKDIR /app

# 3. Copy Code & Configuration
# We copy EVERYTHING (pyproject.toml, api/, etc.) to keep the folder structure
COPY . .

# 4. Install Dependencies
# "pip install ." reads pyproject.toml and installs dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# 5. Set Environment Variable
# Models are now located deeper because we kept the folder structure
ENV ARTIFACT_PATH="/app/api/models_local"

# 6. Expose Port
EXPOSE 8000

# 7. Run Command
# We use "api.main:app" because main.py is inside the 'api' package
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]