# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install Redis, Git, and other dependencies
RUN apt-get update && apt-get install -y \
    redis-server \
    g++ \
    cmake \
    make \
    libstdc++6 \
    git \  
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application code into the container
COPY . /app

# Expose the port the app runs on
EXPOSE 8000

# Create a startup script to run both Redis and your application
RUN echo '#!/bin/bash\nservice redis-server start\nuvicorn main:app --host 0.0.0.0 --port 8000' > /app/start.sh \
    && chmod +x /app/start.sh

# Command to run the startup script
CMD ["/app/start.sh"]
