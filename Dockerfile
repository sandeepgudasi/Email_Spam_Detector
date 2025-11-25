# Use Python 3.11 (compatible with pandas & sklearn)
FROM python:3.11-slim

# Set working dir
WORKDIR /app

# Copy only requirements first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . .

# Expose Flask default port
EXPOSE 5000

# Run your Flask app
CMD ["python", "app.py"]

