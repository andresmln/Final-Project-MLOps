# 1. Base Image
FROM python:3.10-slim-bullseye

# 2. Set Working Directory
WORKDIR /app

# 3. Dependencies (Good caching strategy)
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy API Code
COPY api/ api/

# 5. Copy Artifacts (THIS IS THE MISSING PART) 🚨
# The CI/CD pipeline put the files in 'api/models_local' on the runner.
# We must copy them into the container so the app can find them.
COPY api/models_local/ /app/models_local/

# 6. Set Environment Variable
ENV ARTIFACT_PATH="/app/models_local"

# 7. Expose Port
EXPOSE 7860

# 8. Run Command
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860"]