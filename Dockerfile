# Dockerfile

FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Perintah menjalankan FastAPI menggunakan Uvicorn
# app:app --> nama file: nama instance FastAPI (asumsi file Anda bernama app.py)
CMD exec uvicorn app:app --host 0.0.0.0 --port $PORT