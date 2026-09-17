# Voyage Analytics — Flight Price Prediction API (objective 3)
#
# Build from the PROJECT ROOT (not from inside docker/), since this needs
# access to both api/ and models/, which are siblings of this docker/ folder:
#
#   docker build -f docker/Dockerfile -t voyage-flight-price-api .
#
# Run:
#   docker run -p 5000:5000 voyage-flight-price-api

FROM python:3.10-slim

WORKDIR /app

# Install dependencies first, separately from app code — Docker caches this
# layer, so re-building after just editing app.py won't reinstall packages
# from scratch every time.
COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# Copy the trained model artifacts and the API code.
# app.py locates these via a relative "../models" path from its own file
# location, so this directory layout (models/ and api/ as siblings under
# /app) has to be preserved exactly as it is on disk.
COPY models/ ./models/
COPY api/ ./api/

WORKDIR /app/api

EXPOSE 5000

CMD ["python", "app.py"]