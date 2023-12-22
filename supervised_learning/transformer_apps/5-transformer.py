#!/usr/bin/env python3
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


tf.random.set_seed(1337)  # Set the random seed for reproducibility
# Hyperparameters
batch_size = 64  # How many independent sequences will we process in parallel?
block_size = 96  # What is the maximum context length for predictions?
dropout_rate = 0.1
eval_interval = 100
max_iters = 20000
learning_rate = 0.001
eval_iters = 200
# transformer parameters
n_embd = 128  # Embedding dimension
n_head = 8  # Number of heads in multi-head attention
n_layer = 6  # Number of transformer blocks
# early stopping parameters
min_delta = 0.001
patience = 10

# Define the boundaries and learning rates
step_boundaries = [5000]
lr_values = [0.001, 0.0001]

# read the data
with open(
    "processed_stories.txt", "r", encoding="utf-8"
) as f:
    text = f.read()

# extract what characters are included in the corpus (loaded text)
chars = sorted(list(set(text)))
# find the n value of how many different characters are included in the corpus
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

# use tensoorflow's convert_to_tensor function
data = tf.convert_to_tensor(encode(text), dtype=tf.int64)
# split up the data into train and validation sets
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(data, batch_size, block_size):
    # Generate random indices for batch start positions
    start_indices = tf.random.uniform(
        shape=[batch_size], minval=0, maxval=len(data) - block_size, dtype=tf.int32
    )

    # Prepare batches
    x = tf.TensorArray(dtype=tf.int64, size=batch_size)
    y = tf.TensorArray(dtype=tf.int64, size=batch_size)

    # generate a small batch of data of inputs x and targets y
    for i in tf.range(batch_size):
        start_index = start_indices[i]
        x = x.write(i, data[start_index : start_index + block_size])
        y = y.write(i, data[start_index + 1 : start_index + block_size + 1])

    x = x.stack()
    y = y.stack()
    return x, y


# Estimate loss function (to be further integrated with TensorFlow's training loop)
def estimate_loss(model, data, eval_iters, batch_size, block_size):
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(eval_iters):
            xb, yb = get_batch(data[split], batch_size, block_size)
            logits = model(xb, training=False)
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                yb, logits, from_logits=True
            )
            losses.append(tf.reduce_mean(loss))
        out[split] = tf.reduce_mean(losses).numpy()
    return out


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (size, size)


class Head(layers.Layer):
    """One head of self-attention with causal masking"""

    def __init__(self, head_size):
        super().__init__()
        self.key = layers.Dense(head_size, use_bias=False)
        self.query = layers.Dense(head_size, use_bias=False)
        self.value = layers.Dense(head_size, use_bias=False)
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x, mask=None):
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # Compute attention scores
        wei = tf.matmul(q, k, transpose_b=True)
        wei *= tf.math.rsqrt(tf.cast(tf.shape(k)[-1], tf.float32))

        # Apply masking to the scores
        if mask is not None:
            wei += mask * -1e9

        wei = tf.nn.softmax(wei)
        wei = self.dropout(wei)

        # Weighted aggregation of values
        return tf.matmul(wei, v)

    def get_config(self):
        return super().get_config()


class MultiHeadAttention(layers.Layer):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = [Head(head_size) for _ in range(num_heads)]
        self.proj = layers.Dense(n_embd)
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x):
        seq_len = tf.shape(x)[1]

        mask = create_look_ahead_mask(seq_len)

        head_outputs = [head(x, mask) for head in self.heads]
        out = tf.concat(head_outputs, axis=-1)
        out = self.dropout(self.proj(out))
        return out

    def get_config(self):
        return super().get_config()


class FeedForward(layers.Layer):
    """A simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = keras.Sequential(
            [
                layers.Dense(4 * n_embd),
                layers.ReLU(),
                layers.Dense(n_embd),
                layers.Dropout(dropout_rate),
            ]
        )

    def call(self, x):
        return self.net(x)


class Block(layers.Layer):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# bigram model
class BigramLanguageModel(tf.keras.Model):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super(BigramLanguageModel, self).__init__()
        self.token_embedding_table = layers.Embedding(vocab_size, n_embd)
        self.position_embedding_table = layers.Embedding(block_size, n_embd)
        self.blocks = [Block(n_embd, n_head) for _ in range(n_layer)]
        self.ln_f = layers.LayerNormalization(epsilon=1e-6)
        self.lm_head = layers.Dense(vocab_size)

    def call(self, idx, training=False):
        B = tf.shape(idx)[0]
        T = tf.shape(idx)[1]
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(tf.range(T))  # (T, C)

        x = tok_emb + pos_emb  # Add token and position embeddings
        for block in self.blocks:
            x = block(x)  # Pass through each transformer block
        x = self.ln_f(x)  # Apply final layer norm
        logits = self.lm_head(x)  # Project to vocabulary size

        return logits

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            current_length = tf.shape(idx)[1]
            if current_length > block_size:
                idx = idx[:, -block_size:]  # Keep only the last 'block_size' tokens

            logits = self.call(
                idx, training=False
            )  # Get logits for the current sequence
            logits = logits[:, -1, :]  # Focus only on the last time step
            probs = tf.nn.softmax(
                logits, axis=-1
            )  # Apply softmax to convert logits to probabilities
            idx_next = tf.random.categorical(
                tf.math.log(probs), num_samples=1
            )  # Sample next token
            idx = tf.concat(
                [idx, idx_next], axis=1
            )  # Append sampled token to the sequence
        return idx


from tensorflow.keras.callbacks import EarlyStopping


# Define EarlyStopping callback
early_stopping = EarlyStopping(
    monitor="val_loss",
    min_delta=min_delta,  # Minimum change to qualify as an improvement
    patience=patience,  # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True,  # Restore model weights from the epoch with the best value of the monitored quantity
)

# Variables to keep track of best loss and epochs without improvement
best_loss = float("inf")
epochs_without_improvement = 0

data = {"train": train_data, "val": val_data}

# Instantiate model and optimizer
model = BigramLanguageModel(vocab_size, n_embd, n_head, n_layer, block_size)

# Create the PiecewiseConstantDecay learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=step_boundaries, values=lr_values
)
# Instantiate the optimizer with the learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


@tf.function
def train_step(xb, yb):
    with tf.GradientTape() as tape:
        logits = model(xb, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            yb, logits, from_logits=True
        )
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return tf.reduce_mean(loss)


# Variable to store the best model's weights
best_weights = None

# Training loop
for step in range(max_iters):
    xb, yb = get_batch(train_data, batch_size, block_size)
    loss = train_step(xb, yb)

    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss(model, data, eval_iters, batch_size, block_size)
        print(f"Step {step}: Train Loss: {losses['train']}, Val Loss: {losses['val']}")

        # Update best model if new best loss is found
        current_val_loss = losses["val"]
        if current_val_loss < best_loss - min_delta:
            best_loss = current_val_loss
            best_weights = model.get_weights()  # Save the best model's weights
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered")
                model.set_weights(best_weights)  # Restore the best weights
                break

# Save the best model after training
model_save_path = "/GPT_maker/20ksteps_contextInc"
model.save(model_save_path)
print(f"Best model saved successfully at: {model_save_path}!")


# Text Generation
start_idx = tf.zeros((1, 1), dtype=tf.int64)  # Starting point for generation
generated_idx = model.generate(start_idx, 2000)  # Adjust length as needed
print(decode(generated_idx.numpy()[0]))  # Define your decode function
