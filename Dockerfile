FROM python:3.10-slim

# Install system dependencies for OpenCV, image processing, and headless operation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install CPU PyTorch (lightweight, ~180MB download vs 2.5GB CUDA)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install backend dependencies
COPY requirements-backend.txt .
RUN pip install --no-cache-dir -r requirements-backend.txt

# Copy application source code and models
COPY . .

# Hugging Face Spaces exposes port 7860
ENV PORT=7860
EXPOSE 7860

# Run FastAPI backend with Uvicorn
CMD ["uvicorn", "webapp.backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
