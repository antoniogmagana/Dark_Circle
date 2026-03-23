FROM ros:jazzy

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-rclpy \
    ros-jazzy-rclpy \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip3 install poetry --break-system-packages

# Set working directory
WORKDIR /app

# Copy dependency files and install
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-root

# Copy source
COPY main.py .

# Copy ingestor deployment template
COPY ../../k8s/ingestor-template.yaml .

# Source ROS2 and run
CMD ["/bin/bash", "-c", "source /opt/ros/jazzy/setup.bash && python3 main.py"]
