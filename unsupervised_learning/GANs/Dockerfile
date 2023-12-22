# create image with docker build -t mrialz .
# run container from within alzClassification directory with:
#   docker run -it --rm --gpus all -p 8888:8888 -v $(pwd):/usr/src/app mrialz

# Use an official TensorFlow runtime as a parent image with GPU support
FROM tensorflow/tensorflow:2.6.0-gpu

# Set the working directory in the container
WORKDIR /usr/src/app

RUN python3 -m pip install --upgrade pip

# Copy the current directory contents into the container at /usr/src/app
COPY . /usr/src/app

# Install additional required packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8888 available
EXPOSE 8888

# Set environment variables for NVIDIA CUDA
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.1"
