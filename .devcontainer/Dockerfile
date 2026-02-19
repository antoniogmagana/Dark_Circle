FROM osrf/ros:jazzy-desktop-full

ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# THE FIX: Delete the default 'ubuntu' user that squats on UID 1000
RUN if id -u $USER_UID ; then userdel `id -un $USER_UID` ; fi

# Create a non-root user for development
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install standard dev tools and colcon extensions
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    git \
    nano \
    && rm -rf /var/lib/apt/lists/*

USER $USERNAME
ENV SHELL /bin/bash