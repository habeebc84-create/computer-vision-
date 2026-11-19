# Hugging Face Space (Docker SDK) — builds the frontend and runs the FastAPI backend.
# Stage 1: build the React frontend.
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Python backend (CPU) serving the API + the built frontend.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=7860
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 curl \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple
COPY backend/ ./backend/
COPY run.py ./
COPY --from=frontend-build /app/frontend/dist ./frontend/dist
RUN mkdir -p models dataset/images/all dataset/labels/all dataset/images/train \
    dataset/images/val dataset/labels/train dataset/labels/val \
    outputs/images outputs/videos outputs/reports database uploads
EXPOSE 7860
CMD ["python", "run.py"]
