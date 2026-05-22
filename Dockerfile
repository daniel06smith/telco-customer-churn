# Use a slim Python image — "slim" excludes build tools we don't need,
# keeping the final image size small (~150MB vs ~900MB for the full image).
FROM python:3.11-slim

WORKDIR /app

# --- Layer caching trick ---
# Copy requirements FIRST, install, THEN copy source code.
# Docker caches each step. If you change only your Python files,
# Docker reuses the cached pip install step — much faster rebuilds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the telco_churn package in editable mode
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e .

# Copy app code and the trained model artefacts
COPY app/ app/
COPY models/ models/

# Document which port the app listens on (doesn't publish it — that's
# done with -p when running the container).
EXPOSE 8501

# --server.address=0.0.0.0 is required so the app is reachable from
# outside the container. Without it, Streamlit only listens on localhost
# inside the container and you can never connect to it.
CMD ["streamlit", "run", "app/Predictor.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
