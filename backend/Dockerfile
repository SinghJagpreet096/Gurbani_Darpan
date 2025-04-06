# Use lightweight Python image
FROM python:3.10.12

# Set the working directory
WORKDIR /app

# Copy backend files
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose backend port
EXPOSE 5000

# Start the Flask server
CMD ["python", "app.py"]
