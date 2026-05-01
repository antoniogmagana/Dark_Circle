FROM ros:jazzy

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    ros-jazzy-rclpy \
    ros-jazzy-ament-cmake \
    ros-jazzy-rosidl-default-generators \
    ros-jazzy-rosidl-default-runtime \
    ros-jazzy-builtin-interfaces \
    ros-jazzy-std-msgs \
    ros-jazzy-sensor-msgs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Build context is the inference-engine root.

# Build the ros2_interfaces package so RawSensorReading is importable.
COPY ros2_interfaces /ros2_ws/src/ros2_interfaces
RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash \
    && cd /ros2_ws \
    && colcon build --packages-select ros2_interfaces"

# Source for the fake publisher
COPY src/fake_publisher/main.py .

ENV PYTHONUNBUFFERED=1
CMD ["/bin/bash", "-c", "source /opt/ros/jazzy/setup.bash && source /ros2_ws/install/setup.bash && python3 main.py"]
