FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Copy the trained model
COPY modelHamza/finetuned_model.h5 .


# Expose the port your Flask app is running on
EXPOSE 5000

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_ENV=development

# Set the command to run your Flask app
CMD ["flask", "run", "--host=0.0.0.0"]
