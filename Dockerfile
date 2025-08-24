FROM python:3.11-slim

RUN apt-get update && apt-get install -y git wget build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV UVICORN_WORKERS=1
ENV PORT=8080
EXPOSE 8080

CMD ["python", "main.py"]
