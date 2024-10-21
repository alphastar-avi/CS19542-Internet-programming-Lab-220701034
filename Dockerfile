# Use an official Python runtime as a parent image
FROM python:3.11

# Upgrade pip
RUN pip install --upgrade pip

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install any required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the Python script
CMD ["python", "ollama_langchain.py"]