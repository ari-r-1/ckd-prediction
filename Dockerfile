
# Use official Python 3.10 base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (optional but can help with numpy, pandas, xgboost, etc.)
RUN apt-get update && apt-get install -y build-essential

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Fly.io will use (Gunicorn listens here)
EXPOSE 8080

# Run the app using gunicorn
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:8080"]
