# Start from the latest GPU-enabled TensorFlow image
FROM tensorflow/tensorflow:latest-gpu

# Update the package lists
RUN apt-get update

# Install pip (in case it's not included in the base image)
RUN apt-get install -y python3-pip

# Upgrade pip (optional)
RUN pip install --upgrade pip

# Install scipy
RUN pip install scipy

# Mount the local project directory to the docker image
VOLUME /scripts

# Set working directory in the container to /scripts
WORKDIR /scripts

# By default, run a command that keeps the container running
CMD ["tail", "-f", "/dev/null"]
