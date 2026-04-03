# Expense Audit OpenEnv — Docker image
# Compatible with Hugging Face Spaces (Docker SDK)

FROM python:3.11-slim

# HF Spaces runs as user 1000
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY --chown=appuser:appuser . .

# Create __init__.py files
RUN touch env/__init__.py tasks/__init__.py graders/__init__.py baseline/__init__.py

USER appuser

# HF Spaces expects port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["python", "app.py"]
