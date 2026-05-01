FROM ros:jazzy

RUN apt-get update && apt-get install -y \
    python3-pip \
    ros-jazzy-rclpy \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Build context is the inference-engine root.
# Install Python deps the simple way -- the project pyproject.toml has no
# real runtime dependencies (deps are dev-group only), so just pip-install
# what the discovery node actually needs.
RUN pip3 install --break-system-packages \
        kubernetes \
        pyyaml

# Source for the discovery node
COPY src/discovery/main.py .
COPY src/discovery/whitelist.py .

# Ingestor deployment template (read at runtime to spawn Ingestor pods)
COPY k8s/ingestor-template.yaml .

ENV PYTHONUNBUFFERED=1
CMD ["/bin/bash", "-c", "source /opt/ros/jazzy/setup.bash && python3 main.py"]
