# Configurations for project container
FROM ubuntu:24.04

# Set working directory inside container
WORKDIR /app

# Install system dependencies, clean up disk space, and create virtual environment
RUN apt update -y && \
    apt install -y python3 python3-pip python3-scapy python3-sklearn python3.12-venv && \
    apt install -y net-tools && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m venv venv

ENV PATH="venv/bin:$PATH"

COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy Google API key into image
# -- COPY API KEY HERE --

# Set relative path for key
# -- COPY API KEY PATH here --

# Copy entire project into container
COPY . .

# Execute project when running container
CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]