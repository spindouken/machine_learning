# Convolutions and Pooling

## Table of Contents
- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Task Summaries](#task-summaries)

## Project Overview

This project focuses on implementing various convolution and pooling operations on grayscale and colored images using NumPy. It covers valid and same convolutions, convolutions with custom padding, strided convolutions, and pooling methods.

## Prerequisites

- Python 3.x
- NumPy

## Task Summaries

0. **Valid Convolution**:  
   Implements a valid convolution operation on grayscale images using a specified kernel. The function allows only two for loops and returns a NumPy array of convolved images.

1. **Same Convolution**:  
   Performs a same convolution on grayscale images, ensuring that the output size matches the input size by padding with zeros as needed. The function is limited to two for loops.

2. **Convolution with Padding**:  
   Executes a convolution on grayscale images with custom padding dimensions specified by the user. It uses two for loops and returns the convolved images.

3. **Strided Convolution**:  
   Implements a convolution operation on grayscale images with adjustable stride and padding options, allowing for 'same' or 'valid' convolutions. The function is restricted to two for loops.

4. **Convolution with Channels**:  
   Performs a convolution on images with multiple channels, accommodating various padding and stride settings. The function returns convolved images and utilizes two for loops.

5. **Multiple Kernels**:  
   Executes a convolution on images using multiple kernels, returning convolved images. The function supports 'same' and 'valid' padding and is limited to three for loops.

6. **Pooling**:  
   Implements pooling operations (max or average) on images, specifying kernel size and stride. The function returns pooled images and adheres to the two for loop restriction.
