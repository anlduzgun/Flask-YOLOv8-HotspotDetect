# Use the official Python image as a base image
FROM python:3


# Set the environment variable
ENV PYTHONBUFFERED=1


# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .


# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Expose port 5000 for the Flask application
EXPOSE 8000

# Run the Flask application
CMD ["python3", "app.py"]

