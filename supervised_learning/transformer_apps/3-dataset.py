#!/usr/bin/env python3
"""
update the class Dataset that loads and preps a dataset for machine translation
"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """
    loads and preps a dataset for machine translation
    """

    def __init__(self, batch_size, max_len):
        """
        class constructor

        contains the instance attributes:
        data_train, which contains the ted_hrlr_translate/pt_to_en
            tf.data.Dataset train split, loaded as_supervided
        data_valid, which contains the ted_hrlr_translate/pt_to_en
            tf.data.Dataset validate split, loaded as_supervided
        tokenizer_pt is the Portuguese tokenizer
            created from the training set
        tokenizer_en is the English tokenizer
            created from the training set
        """
        # load dataset
        dataset, metadata = tfds.load(
            "ted_hrlr_translate/pt_to_en", with_info=True, as_supervised=True
        )
        self.metadata = metadata
        self.data_train = dataset["train"]
        self.data_valid = dataset["validation"]

        # tokenization
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

        # data pipeline for training data
        self.data_train = self.data_train.map(self.tf_encode)
        # filter out all examples that have either sentence with more than max_len tokens
        self.data_train = self.data_train.filter(
            lambda pt, en: tf.logical_and(
                tf.size(pt) <= max_len, tf.size(en) <= max_len
            )
        )
        # cache the dataset to increase performance
        self.data_train = self.data_train.cache()

        # shuffle the dataset
        self.data_train = self.data_train.shuffle(metadata.splits["train"].num_examples)

        # split dataset into padded batches at size of arg: batch_size
        self.data_train = self.data_train.padded_batch(
            batch_size, padded_shapes=([None], [None])
        )
        # prefetch the dataset using experimental.AUTOTUNE to increase performance
        self.data_train = self.data_train.prefetch(tf.data.experimental.AUTOTUNE)

        # data pipeline for validation data
        self.data_valid = self.data_valid.map(self.tf_encode)
        self.data_valid = self.data_valid.filter(
            lambda pt, en: tf.logical_and(
                tf.size(pt) <= max_len, tf.size(en) <= max_len
            )
        )
        # split validation dataset into batches at size of arg: batch_size
        self.data_valid = self.data_valid.padded_batch(
            batch_size, padded_shapes=([None], [None])
        )

    def tokenize_dataset(self, data):
        """
        creates sub-word tokenizers for our dataset

        data is a tf.data.Dataset whose examples are formatted as a tuple (pt, en)
            pt is the tf.Tensor containing the Portuguese sentence
            en is the tf.Tensor containing the corresponding English sentence
        The maximum vocab size should be set to 2**15

        Returns: tokenizer_pt, tokenizer_en
            tokenizer_pt is the Portuguese tokenizer
            tokenizer_en is the English tokenizer
        """
        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2**15
        )
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2**15
        )
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        encodes a translation into tokens
        pt is the tf.Tensor containing the Portuguese sentence
        en is the tf.Tensor containing the corresponding English sentence
        the tokenized sentences should include the start and end of sentence tokens
        the start token should be indexed as vocab_size
        the end token should be indexed as vocab_size + 1
        Returns: pt_tokens, en_tokens
            pt_tokens is a np.ndarray containing the Portuguese tokens
            en_tokens is a np.ndarray. containing the English tokens
        """
        pt_tokens = (
            [self.tokenizer_pt.vocab_size]
            + self.tokenizer_pt.encode(pt.numpy())
            + [self.tokenizer_pt.vocab_size + 1]
        )
        en_tokens = (
            [self.tokenizer_en.vocab_size]
            + self.tokenizer_en.encode(en.numpy())
            + [self.tokenizer_en.vocab_size + 1]
        )
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        acts as a tensorflow wrapper for the encode instance method
        """
        result_pt, result_en = tf.py_function(
            self.encode, [pt, en], [tf.int64, tf.int64]
        )
        result_pt.set_shape([None])
        result_en.set_shape([None])
        return result_pt, result_en
