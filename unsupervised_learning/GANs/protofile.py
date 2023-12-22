#!/usr/bin/env python3
import wandb
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose, Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, ReLU, LeakyReLU, Input
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, Flatten, Reshape
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_data(data_dir, batch_size=32):
    """
    Load the Alzheimer MRI dataset from a specified directory
    that now includes synthetic images.

    Args:
        data_dir (str): directory where the dataset is stored
        batch_size (int): batch size for the data generator
            will default to 32 if not specified when calling the function

    Returns:
        trainingGenerator, validationGenerator: data generators for training and validation sets
    """
    datagen = ImageDataGenerator(validation_split=0.2)  # 20% data for validation

    trainingGenerator = datagen.flow_from_directory(
        data_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validationGenerator = datagen.flow_from_directory(
        data_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return trainingGenerator, validationGenerator


def build_generator(noise_dim=100):
    model = Sequential()
    model.add(Dense(8 * 8 * 128, input_dim=noise_dim))
    model.add(Reshape((8, 8, 128)))

    # Up-sample to 16x16
    model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(ReLU())

    # Up-sample to 32x32
    model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(ReLU())

    # Up-sample to 64x64
    model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(ReLU())

    # Up-sample to 128x128
    model.add(Conv2DTranspose(3, kernel_size=4, strides=2, padding="same", activation="tanh"))

    return model

def build_discriminator(input_shape=(128, 128, 3)):
    model = Sequential()

    model.add(Conv2D(64, kernel_size=4, strides=2, padding="same", input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(128, kernel_size=4, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(128, kernel_size=4, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))

    return model



def save_imgs(generator, epoch, noise_dim=100):
    r, c = 5, 5  # grid size
    noise = np.random.normal(0, 1, (r * c, noise_dim))
    gen_imgs = generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, :])
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(f"/content/drive/MyDrive/MRIalz/GANgen/epoch_{epoch}.png")
    plt.close()


def gan_training(generator, discriminator, trainingGenerator, use_wandb=True, noise_dim=100, epochs=100, batch_size=32, patience=3):
    if use_wandb:
        wandb.init(project='gan_for_alzheimers', name='GAN-test')

    best_g_loss = float('inf')  # Initialize the best loss as infinity
    epochs_without_improvement = 0

    # Bayesian Optimization could be integrated here for hyperparameter tuning

    # Using Adam optimizer and binary cross-entropy loss as they generally work well for GANs
    discriminator.compile(optimizer=Adam(0.00002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

    # Placeholder for noise input to the generator
    noise = Input(shape=(noise_dim,))
    generated_image = generator(noise)

    # For the combined model, we will only train the Generator
    discriminator.trainable = False

    # The Discriminator will evaluate the generated image
    validity = discriminator(generated_image)

    # Combined model: stack the generator and discriminator
    combined = Model(noise, validity)
    combined.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

    for epoch in range(epochs):
        # Get a batch of real images
        real_imgs, _ = trainingGenerator.next()
        real_imgs = (real_imgs - 127.5) / 127.5
        print(f"Epoch {epoch}, Real Image Shape: {real_imgs.shape}")  # Debugging line


        # Skip incomplete batches
        if real_imgs.shape[0] != batch_size:
            print(f"Skipping epoch {epoch} due to incomplete batch.")
            continue

        # Generate a batch of fake images
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        fake_imgs = generator.predict(noise)

        # Labels for real and fake images
        real_y = np.ones((batch_size, 1))
        fake_y = np.zeros((batch_size, 1))

        # Train the Discriminator
        d_loss_real = discriminator.train_on_batch(real_imgs, real_y)
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_y)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train the Generator
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        valid_y = np.ones((batch_size, 1))  # The generator tries to make the discriminator classify these as real

        # Training the combined model (only updates the Generator)
        g_loss = combined.train_on_batch(noise, valid_y)

        # Check for early stopping
        if g_loss < best_g_loss:
            best_g_loss = g_loss
            epochs_without_improvement = 0  # Reset the counter
        else:
            epochs_without_improvement += 1  # Increment the counter

        # Stop training if the counter exceeds patience
        if epochs_without_improvement > patience:
            print("Early stopping triggered. Stopping training.")
            break

        if use_wandb:
            wandb.log({
                "Epoch": epoch,
                "Discriminator Loss": d_loss[0],
                "Discriminator Accuracy": 100 * d_loss[1],
                "Generator Loss": g_loss
            })

        # Save model weights (optional, can be done periodically)
        if epoch % 100 == 0:
            # generator.save_weights(f"./generator_epoch_{epoch}.h5")
            # # discriminator.save_weights(f"./discriminator_epoch_{epoch}.h5")
            # wandb.save(f"./generator_epoch_{epoch}.h5")
            # wandb.save(f"./discriminator_epoch_{epoch}.h5")
            save_imgs(generator, epoch, noise_dim)


        # Logging (WandB could be integrated here for experiment tracking)
        print(f"{epoch} [D loss: {d_loss[0]} | D Accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")


# Main script
if __name__ == "__main__":
    # set to True or False depending on if using wandb
    use_wandb = False
    if use_wandb:
        wandb.init(project='gan_for_alzheimers', name='GAN-Main-Run')
    generator = build_generator()
    discriminator = build_discriminator()
    trainingGenerator, _ = load_data('/content/drive/MyDrive/MRIalz/Dataset')

    if use_wandb:
        wandb.config.update({"Generator Architecture": generator.summary()})
        wandb.config.update({"Discriminator Architecture": discriminator.summary()})

    gan_training(generator, discriminator, trainingGenerator, use_wandb=use_wandb)
