# Use a lightweight Python image
FROM python:3.9-slim

# Set the folder inside the container where our code will live
WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy the requirements file and install libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the stopword data for our preprocessing function
RUN python -m nltk.downloader stopwords

# Copy our source code and our trained model into the container
COPY src/ /app/src/
COPY models/ /app/models/

# Tell Docker that the container will use port 8000
EXPOSE 8000

# The command to start our API when the container runs
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]