FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir flask==3.0.3 gunicorn==22.0.0

COPY app.py .

EXPOSE 9999

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:9999", "--workers", "2", "--timeout", "30"]
