FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.128.0 \
    uvicorn[standard]==0.40.0 \
    requests==2.32.3 \
    redis==5.2.1 \
    python-dotenv==1.0.1 \
    pydantic==2.12.0

COPY *.py /app/

ENV PORT=8300 HOST=0.0.0.0 LOG_LEVEL=INFO PYTHONUNBUFFERED=1

EXPOSE 8300

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8300/healthz', timeout=3)" || exit 1

CMD ["python3", "search_router_service.py"]
