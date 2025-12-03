# Configurations for project container
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Install all project dependencies
COPY requirements.txt .

RUN pip3 install --upgrade pip3 && pip3 install --no-cache-dir -r requirements.txt

# Copy entire project into container
COPY . .

# Execute project when running container
CMD ["python3", "main.py"]
