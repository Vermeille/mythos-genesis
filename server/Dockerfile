# Use the official Python image from Docker Hub
FROM python:3.12

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

# Create directory for the app user
WORKDIR $APP_HOME

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port Uvicorn will run on
EXPOSE 8000

# Command to run the FastAPI application
CMD ["fastapi", "dev", "--host", "0.0.0.0", "main.py"]

