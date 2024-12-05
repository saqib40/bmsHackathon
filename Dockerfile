# Use a base image with Python
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY backend/requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code
COPY backend/ .

# Copy the trained model
COPY backend/modelHamza/finetuned_model.h5 .

# Copy the frontend code
COPY frontend/ ./frontend/

# Expose the port your Flask app is running on
EXPOSE 5000

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_ENV=development

# Set the command to run your Flask app
CMD ["flask", "run", "--host=0.0.0.0"]
