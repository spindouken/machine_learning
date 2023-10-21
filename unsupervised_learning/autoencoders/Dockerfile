# Use an official TensorFlow runtime as a parent image with GPU support
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . /usr/src/app

# Install additional required packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Set environment variables for NVIDIA CUDA
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.1"

# Run the optimization script when the container launches
CMD ["python", "6-bayes_opt.py"]
