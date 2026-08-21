FROM python:3.11-slim

ENV https_proxy="http://163.116.128.80:8080"
ENV http_proxy="http://163.116.128.80:8080"

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

ENV PORT=8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
