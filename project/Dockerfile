FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir --force-reinstall scikit-learn==1.8.0 numpy==2.2.4 pandas==2.2.3

COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY configs/ ./configs/

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]