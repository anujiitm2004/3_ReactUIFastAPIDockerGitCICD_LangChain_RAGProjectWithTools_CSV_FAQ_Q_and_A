# Base image
FROM python:3.10.2-slim

# Set working directory
WORKDIR /myapp4

# Install only required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies separately (to leverage caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose the FastAPI default port
EXPOSE 5000

# CMD to run FastAPI
CMD ["uvicorn", "FastAPI_server:app", "--host", "0.0.0.0", "--port", "5000"]