#!/usr/bin/env python3
"""
Create the class Dataset that loads and preps a dataset for machine translation using tensorflow_datasets
"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """
    loads and preps a dataset for machine translation
    """

    def __init__(self):
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

        what is a tf.data.Dataset?
        A tf.data.Dataset is a sequence of elements that can be processed in a streaming fashion.
        The elements of a dataset are tensors.

        what is a streaming fashion?
        A streaming fashion is a way of processing data one element at a time.

        how does tfds work?
        tfds works by downloading the dataset to disk and then loading it into memory.
        The first time you use a dataset, it is downloaded to your disk.
        The next time it is used, it is loaded from disk.

        what are some commands when working with tfds?
        tfds.list_builders() - lists all datasets
        tfds.load() - loads a dataset
        tfds.as_numpy() - converts a tf.data.Dataset into an iterable that yields NumPy arrays
        tfds.as_supervised() - loads a dataset in supervised mode
        """
        dataset, metadata = tfds.load(
            "ted_hrlr_translate/pt_to_en", with_info=True, as_supervised=True
        )
        self.metadata = metadata
        self.data_train = dataset["train"]
        self.data_valid = dataset["validation"]
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

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
