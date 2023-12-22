#!/usr/bin/env python3
"""
creates all masks for training/validation
"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


def create_masks(inputs, target):
    """
    creates all masks for training/validation

    inputs is a tf.Tensor of shape (batch_size, seq_len_in) that contains
        the input sentence
    target is a tf.Tensor of shape (batch_size, seq_len_out) that contains
        the target sentence

    Returns: encoder_mask, combined_mask, decoder_mask
        encoder_mask is the tf.Tensor padding mask of shape
            (batch_size, 1, 1, seq_len_in) to be applied in the encoder
        combined_mask is the tf.Tensor of shape
            (batch_size, 1, seq_len_out, seq_len_out) used in the 1st
                attention block in the decoder to pad and mask future tokens
                in the input received by the decoder. It takes the maximum
                between a lookahead mask and the decoder target padding mask.
        decoder_mask is the tf.Tensor padding mask of shape
            (batch_size, 1, 1, seq_len_in) used in the 2nd attention block
                in the decoder.
    """
    # get the batch size and sequence lengths from the input and target tensors
    batch_size, seq_len_out = target.shape
    batch_size, seq_len_in = inputs.shape

    # create tf.Tensor encoder padding mask
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    # expand dimensions to add the padding to the attention logits
    encoder_mask = encoder_mask[
        :, tf.newaxis, tf.newaxis, :
    ]  # (batch_size, 1, 1, seq_len_in)

    # create decoder tf.Tensor decoder padding mask (similar to encoder mask)
    #   used in the second attention block (encoder-decoder attention)
    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[
        :, tf.newaxis, tf.newaxis, :
    ]  # (batch_size, 1, 1, seq_len_in)

    # create look-ahead mask to prevent the model from seeing future tokens
    #   look-ahead mask and decoder target padding mask for the first attention block (self-attention)
    lookAheadMask = 1 - tf.linalg.band_part(tf.ones((seq_len_out, seq_len_out)), -1, 0)

    # create decoder target padding mask
    # Identify padding tokens in target and expand dimensions
    decoderTargetPaddingMask = tf.cast(tf.math.equal(target, 0), tf.float32)
    decoderTargetPaddingMask = decoderTargetPaddingMask[
        :, tf.newaxis, tf.newaxis, :
    ]  # (batch_size, 1, 1, seq_len_out)

    # create combined mask for the first attention block in the decoder
    #   combines the look-ahead mask and the decoder target padding mask
    combined_mask = tf.maximum(
        decoderTargetPaddingMask, lookAheadMask
    )  # (batch_size, 1, seq_len_out, seq_len_out)
    return encoder_mask, combined_mask, decoder_mask
