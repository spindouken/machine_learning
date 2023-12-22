#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# Data Preparation
(train_images, _), (_, _) = mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize to [-1, 1]

def build_initial_generator():
    model = tf.keras.Sequential([
        # Fully connected layer, followed by reshape
        layers.Dense(4*4*256, use_bias=False, input_shape=(100,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((4, 4, 256)),  # Reshape it to a 4x4 image

        # Transposed Convolution
        layers.Conv2DTranspose(128, (4, 4), strides=(1, 1), padding='valid', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        # Output layer to produce 4x4 grayscale image
        layers.Conv2DTranspose(1, (4, 4), strides=(1, 1), padding='valid', use_bias=False, activation='tanh')
    ])
    return model


def build_initial_discriminator():
    model = tf.keras.Sequential([
        # Convolution Layer
        layers.Conv2D(128, (4, 4), strides=(1, 1), padding='valid', input_shape=[4, 4, 1]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        # Flatten and output layer
        layers.Flatten(),
        layers.Dense(1)  # Output layer
    ])
    return model

# Utility Function to Generate and Save Images
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

def add_layers_to_generator(model):
    """
    Add new layers to the generator.
    """
    print("Before modification:")
    model.summary()
    new_model = tf.keras.Sequential()
    for layer in model.layers[:-1]:  # Exclude the last layer
        new_model.add(layer)

    # Add an intermediate Conv2DTranspose layer that matches the last layer's filter dimension (128)
    new_model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    new_model.add(layers.BatchNormalization())
    new_model.add(layers.LeakyReLU())

    new_model.add(model.layers[-1])  # Add the last layer back

    print("After modification:")
    model.summary()
    return new_model

def add_layers_to_discriminator(model):
    """
    Add new layers to the discriminator.
    """
    new_model = tf.keras.Sequential()
    for layer in model.layers[:-3]:  # Exclude the last Flatten and Dense layers
        new_model.add(layer)
        
    # New Conv2D layer
    new_model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same'))
    new_model.add(layers.LeakyReLU())
    new_model.add(layers.Dropout(0.3))

    # Flatten layer
    new_model.add(layers.Flatten())

    # New Dense layer with 128 units to match last layer's input
    new_model.add(layers.Dense(128, activation='relu'))

    # Add last Dense layer back
    new_model.add(model.layers[-1])
    
    return new_model



def calculate_fadein_alpha(current_epoch, total_epochs):
    """
    Calculate the fade-in alpha value based on the current epoch and total epochs.
    """
    return min(1, current_epoch / float(total_epochs))


from tensorflow.keras import backend as K

class CustomGenerator(tf.keras.Model):
    def __init__(self, generator):
        super(CustomGenerator, self).__init__()
        self.generator = generator

    def call(self, inputs):
        x = self.generator(inputs)
        return x

    def penultimate_output(self, inputs):
        for layer in self.generator.layers[:-1]:
            inputs = layer(inputs)
        return inputs


def downscale_images(images):
    """
    Downscale images by 2x2.
    """
    return tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(images)

# Training Step
def train_step(images, step, fadein_alpha):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        # Apply fade-in to the generated images if alpha < 1
        if fadein_alpha < 1:
            # Get the output from the penultimate layer
            penultimate_output = generator.penultimate_output(noise)
            generated_images = fadein_alpha * generated_images + (1 - fadein_alpha) * penultimate_output
            
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        # Apply fade-in to the discriminator outputs if alpha < 1
        if fadein_alpha < 1:
            # Get the downscaled version of the input images
            downscaled_images = downscale_images(images)
            old_disc_output = discriminator(downscaled_images, training=True)
            real_output = fadein_alpha * real_output + (1 - fadein_alpha) * old_disc_output
 
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

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 50  # Increased epochs for demonstration
noise_dim = 100
num_examples_to_generate = 16
BUFFER_SIZE = 60000
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Loss and Optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Initialize Generator and Discriminator
generator = CustomGenerator(build_initial_generator())
discriminator = build_initial_discriminator()

# Initialize fade-in related variables
fadein_duration = 5  # Number of epochs over which to apply fade-in for a new layer
current_fadein_epoch = 0  # To keep track of fade-in duration for the current layer

# Initialize tqdm
pbar = tqdm(range(EPOCHS))

# Main Training Loop
for epoch in pbar:
    if epoch % fadein_duration == 0:
        generator = add_layers_to_generator(generator)
        discriminator = add_layers_to_discriminator(discriminator)
        current_fadein_epoch = 0
    # Calculate fade-in alpha for the current epoch
    fadein_alpha = calculate_fadein_alpha(current_fadein_epoch, fadein_duration)
    
    for step, image_batch in enumerate(train_dataset):
        gen_loss, disc_loss, gen_acc, disc_acc = train_step(image_batch, step, fadein_alpha)
        train_step(image_batch, step)
    generate_and_save_images(generator, epoch, tf.random.normal([num_examples_to_generate, noise_dim]))
    
    # Logic to decide when to grow networks
    if epoch % 10 == 0 and epoch > 0:  # Example: grow every 10 epochs
        add_layer_to_generator()
        add_layer_to_discriminator()
    
    current_fadein_epoch += 1
