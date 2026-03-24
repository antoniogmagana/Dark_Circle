FROM ros:jazzy

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip3 install poetry --break-system-packages

# Set working directory
WORKDIR /app

# Copy dependency files and install
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-root

# Copy and install shared proto package
COPY ../../inference-protos /inference-protos
RUN pip install /inference-protos --break-system-packages

# Copy source
COPY main.py ./

# Run
CMD ["python3", "main.py"]
