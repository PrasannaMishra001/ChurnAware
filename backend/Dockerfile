FROM python:3.12-slim

WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY src src
COPY backend backend
COPY models models
COPY data/blinkit_products.csv data/blinkit_products.csv

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
