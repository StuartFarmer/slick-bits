FROM python:3.12-slim
RUN pip install --no-cache-dir "numpy>=2,<3"
WORKDIR /app
COPY problems.py worker.py /app/qube/
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
USER 65534:65534
ENTRYPOINT ["python", "-m", "qube.worker"]
