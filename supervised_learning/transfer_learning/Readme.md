# Transfer Learning

## Supplementary Medium Article
https://medium.com/@masonthecount/classifying-the-cifar-10-dataset-with-transfer-learning-and-tensorflow-keras-237f4cbe5fd9

## Project Summary

This project involves training a convolutional neural network (CNN) using transfer learning to classify images from the CIFAR-10 dataset. It emphasizes leveraging pre-trained Keras applications, specifically MobileNetV2, to enhance classification performance while minimizing training time. 

The project includes two main components: 
1. **Training a base model** (`lilTinyNet`) that utilizes MobileNetV2, with a focus on preprocessing data, optimizing learning rates, and preventing overfitting through early stopping and model checkpointing.
2. **Fine-tuning the model** (`lilTinyNet_finetuner`) to improve accuracy further by unfreezing some layers and applying data augmentation techniques, followed by additional training to refine the model's performance.

### Key Features
- **Model Evaluation**: The project includes scripts to evaluate both the base and fine-tuned models on the CIFAR-10 dataset, providing metrics such as accuracy and loss.
- **Model Visualization**: Tools for visualizing model architecture are provided, allowing users to inspect the structure and layers of both models.
- **Learning Rate Scheduling**: A learning rate scheduler is implemented to adjust the learning rate dynamically during training, which helps improve convergence.
- **Data Preprocessing**: Custom preprocessing functions ensure that images are appropriately scaled and formatted for the MobileNetV2 model.

### Included Files
- **`Dockerfile`**: Sets up a Docker environment to facilitate reproducible experiments, leveraging TensorFlow with GPU support.
- **`evaluator_lilTinyNet&finetuned.py`**: A script that loads both the base and fine-tuned models, evaluates them on the CIFAR-10 dataset, visualizes their architectures, and displays predictions.
- **`lilTinyNet.py`**: Contains the definition of the base model architecture, data preprocessing function, and implements early stopping and model checkpointing during training.
- **`lilTinyNet_finetuner.py`**: Extends the training of `lilTinyNet` by applying data augmentation, unfreezing the last few layers for fine-tuning, and adjusting the learning rate for improved performance.
- **`megaTest.py`**: A consolidated evaluator for both models, showcasing their architectures and providing evaluation metrics.
