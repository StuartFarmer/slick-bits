FROM python:3.12-slim
RUN pip install --no-cache-dir numpy==2.2.6
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
USER 65534:65534
