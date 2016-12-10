#! /usr/bin/env python
# imports are done here

import tensorflow as tf
from tensorflow.contrib import learn

import numpy as np

# Importing the helper functions from the data_helpers
from bundleBot import data_helpers

# defining flags for the tensorflow data loading...

tf.flags.DEFINE_string("positive_data_file", "rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "rt-polarity.neg", "Data source for the positive data.")
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

# making things more readable
FLAGS = tf.flags.FLAGS

# Loading the data

print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

# Get the maximum length of the sentences
max_document_length = max([len(x.split(" ")) for x in x_text])
print(max_document_length)
print(x_text[0])

#creating the processor for the word2vec
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

# Convert the given text into the vectors
x = np.array(list(vocab_processor.fit_transform(x_text)))

# creating the random seed for numpy
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
print(shuffle_indices)

# assigning whatever array we generated from the random generator and getting the random list
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

print(x_shuffled[0])

# dividing the given data-set into test and train set
# TODO: use tfid

dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))



# making helper functions
# TODO: put this in a separate file and then import

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

