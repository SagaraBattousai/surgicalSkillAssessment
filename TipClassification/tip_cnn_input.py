from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import random
import os
from functools import partial

TRAINING_INPUTS = "data" + os.sep + "trainImages"

TRAINING_LABELS = "data" + os.sep + "trainLabels"

EVAL_INPUTS = "data" + os.sep + "evalImages"

EVAL_LABELS = "data" + os.sep + "evalLabels"

IMAGE_WIDTH = 320

IMAGE_HEIGHT = 180

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 8000

NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 174

EPOCH_SIZE = 750

BATCH_SIZE = 128

SHUFFLE_BUFFER_SIZE = BATCH_SIZE

EPOCH_PER_DECAY = 384.0

def _decode_labels(label):

    x = tf.feature_column.numeric_column('x', shape=(1,2), 
                                         default_value=[[0,0]],dtype=tf.int64)
    
    y = tf.feature_column.numeric_column('y', shape=(1,2), 
                                         default_value=[[0,0]],dtype=tf.int64)
    
    width = tf.feature_column.numeric_column('width', shape=(1,2), 
                                         default_value=[[0,0]],dtype=tf.int64)
    
    height = tf.feature_column.numeric_column('height', shape=(1,2), 
                                         default_value=[[0,0]],dtype=tf.int64)

    columns = [x, y, width, height]
    
    pf = tf.parse_single_example(label, 
            features=tf.feature_column.make_parse_example_spec(columns))

    label_tensor = tf.feature_column.input_layer(pf, columns)
    
    return label_tensor

def _decode_images(filename):
    image_string = tf.read_file(filename)
    decoded_image = tf.image.decode_image(image_string, channels=1)

    float_image = tf.image.per_image_standardization(decoded_image)

    float_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    return float_image
    
def build_double_dataset(filename, label):
    image_dataset = tf.data.Dataset.from_tensors(filename)
    image_dataset = image_dataset.map(_decode_images)

    label_dataset = tf.data.TFRecordDataset(label)
    label_dataset = label_dataset.map(_decode_labels)

    return tf.data.Dataset.zip((image_dataset, label_dataset))

def setup_data(input_dir, label_dir, examples_to_setup):

    filenames_list = []
    labels_list = []

    for i in range(1, examples_to_setup + 1):
        filenames_list.append('{}{}frame_{}.jpg'.format(input_dir,
                                                        os.sep,
                                                        i))
    
        label_name = '{}{}frame_{}.tfrecord'.format(label_dir,
                                                    os.sep,
                                                   i)
        labels_list.append(label_name)
    
    filenames = tf.constant(filenames_list)

    labels = tf.constant(labels_list)

    return filenames, labels

def get_train_data():
    filenames, labels = setup_data(TRAINING_INPUTS, TRAINING_LABELS,
                                   NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.flat_map(build_double_dataset)
    #dataset = dataset.map(randomlyFlipDataset)
    dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
    dataset = dataset.repeat(EPOCH_SIZE)
    dataset = dataset.batch(BATCH_SIZE)

    return dataset

def get_eval_data():
    filenames, labels = setup_data(EVAL_INPUTS, EVAL_LABELS, 
                                   NUM_EXAMPLES_PER_EPOCH_FOR_EVAL)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.flat_map(build_double_dataset)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(BATCH_SIZE)

    return dataset

