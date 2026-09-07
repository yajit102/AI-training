# FlowPredict TAT model — deployable pipeline image.
#
# Build:  docker build -f deploy/Dockerfile -t flowpredict:latest .
# Run:    docker run --rm -v "$(pwd)/output:/app/output" flowpredict:latest --n-cases 8000
#
# The container runs the training/inference pipeline as a one-shot job
# (this is a batch model pipeline, not a web service) and writes its
# diagnostics report to stdout, which is redirectable via the compose
# service below.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
COPY docs ./docs

RUN useradd --create-home --uid 1000 flowpredict \
    && chown -R flowpredict:flowpredict /app
USER flowpredict

ENTRYPOINT ["python", "scripts/train_flowpredict_tat_model.py"]
CMD ["--n-cases", "8000"]
