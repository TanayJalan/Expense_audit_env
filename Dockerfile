FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chown -R appuser:appuser /app

RUN mkdir -p env tasks graders baseline data && touch env/__init__.py tasks/__init__.py graders/__init__.py baseline/__init__.py data/__init__.py

USER appuser

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["python", "/app/app.py"]
