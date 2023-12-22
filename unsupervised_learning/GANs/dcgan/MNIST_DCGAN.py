#!/usr/bin/env python3
# Import necessary modules
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tqdm import tqdm
import os
import matplotlib.pyplot as plt


(train_images, _), (_, _) = mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
# Normalize to [-1, 1]
train_images -= 127.5
train_images /= 127.5 

from tensorflow.keras import layers

def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(7*7*256, use_bias=False, input_shape=(100,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((7, 7, 256)),
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model

def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model



import matplotlib.pyplot as plt
import os

# Create a directory to save the images
os.makedirs("/images", exist_ok=True)

# Function to generate and save images
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    predictions = (predictions + 1) / 2.0  # Rescale to [0, 1]

    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(f'MNIST_DCGAN/images/image_at_epoch_{epoch:04d}.png')
    plt.show()


# Initialize Weights and Biases
use_wandb = False  # Set this to False if you don't want to use wandb
if use_wandb:
    import wandb  # Import wandb here to make it optional
    wandb.init(project='mnist_dcgan', name='dcgan_run')

# Hyperparameters and other settings (Code remains the same, so not shown here)

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 5
noise_dim = 100
num_examples_to_generate = 16

# Prepare the dataset
BUFFER_SIZE = 60000
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Add prefetch to optimize the data pipeline
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

# Initialize generator and discriminator
generator = build_generator()
discriminator = build_discriminator()

# Loss and optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


# Initialize tqdm with the total number of epochs
pbar = tqdm(range(EPOCHS))

# Training Loop (Code remains the same, so not shown here)

# Training step
def train_step(images, step):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        disc_loss = cross_entropy(tf.ones_like(real_output), real_output) + cross_entropy(tf.zeros_like(fake_output), fake_output)
        
    # Calculate accuracies
    real_accuracy = tf.reduce_mean(tf.cast(tf.math.greater_equal(real_output, 0.5), tf.float32))
    fake_accuracy = tf.reduce_mean(tf.cast(tf.math.less(fake_output, 0.5), tf.float32))
    disc_accuracy = (real_accuracy + fake_accuracy) / 2.0
    gen_accuracy = tf.reduce_mean(tf.cast(tf.math.greater_equal(fake_output, 0.5), tf.float32))
    
    # Apply gradients
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss, gen_accuracy, disc_accuracy

# Create a random noise vector for image generation
test_input = tf.random.normal([num_examples_to_generate, noise_dim])
# Initialize metrics to track loss and accuracy
cumulative_gen_loss = 0.0
cumulative_disc_loss = 0.0
cumulative_gen_accuracy = 0.0
cumulative_disc_accuracy = 0.0
num_steps = 0


# Training loop with tqdm and wandb
for epoch in pbar:
    # Reset cumulative metrics for each epoch
    cumulative_gen_loss = 0.0
    cumulative_disc_loss = 0.0
    cumulative_gen_accuracy = 0.0
    cumulative_disc_accuracy = 0.0
    num_steps = 0
    
    for step, image_batch in enumerate(train_dataset):
        # Training
        gen_loss, disc_loss, gen_acc, disc_acc = train_step(image_batch, step)
        cumulative_gen_loss += gen_loss
        cumulative_disc_loss += disc_loss
        cumulative_gen_accuracy += gen_acc
        cumulative_disc_accuracy += disc_acc
        num_steps += 1
    
    # Calculate average metrics
    avg_gen_loss = cumulative_gen_loss / num_steps
    avg_disc_loss = cumulative_disc_loss / num_steps
    avg_gen_accuracy = cumulative_gen_accuracy / num_steps
    avg_disc_accuracy = cumulative_disc_accuracy / num_steps
    
    # Log average metrics for the epoch
    if use_wandb:
        wandb.log({"epoch": epoch, "avg_gen_loss": avg_gen_loss, "avg_disc_loss": avg_disc_loss, "avg_gen_accuracy": avg_gen_accuracy, "avg_disc_accuracy": avg_disc_accuracy})
    else:
        print(f"Epoch {epoch+1}, Avg Gen Loss: {avg_gen_loss}, Avg Disc Loss: {avg_disc_loss}, Avg Gen Accuracy: {avg_gen_accuracy}, Avg Disc Accuracy: {avg_disc_accuracy}")
    # Image generation and other code...
    if epoch % 1 == 0:
        generate_and_save_images(generator, epoch, test_input)
    
    # Update tqdm description
    pbar.set_description(f"Epoch {epoch+1}, Avg Gen Loss: {avg_gen_loss}, Avg Disc Loss: {avg_disc_loss}")
