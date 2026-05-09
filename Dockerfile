FROM python:3.11-slim

LABEL maintainer="ClimateShield AI"
LABEL description="AI Early Warning System for Climate-Related Child Health Risks"
LABEL license="MIT"

# System dependencies for geospatial libraries
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create data directories
RUN mkdir -p data/raw data/processed models/saved

# Run as non-root user
RUN useradd -m -u 1000 climateshield && chown -R climateshield:climateshield /app
USER climateshield

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
