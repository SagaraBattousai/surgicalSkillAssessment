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

CHECKPOINT_DIR = "surgical_lnr_checkpoints"

PREDICTION_CHECKPOINT_DIR = CHECKPOINT_DIR

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 8000

NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 174

EPOCH_SIZE = 750

BATCH_SIZE = 128

SHUFFLE_BUFFER_SIZE = BATCH_SIZE

EPOCH_PER_DECAY = 384.0

unzip = lambda z: list(zip(*z))

def cnn_model_fn(features, labels, mode):
    #Input layer
    input_layer = tf.reshape(tf.cast(features, tf.float32), [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    #Convolutional Layer #1
    conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

    #Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[4, 4],
            strides=4)

    #Normalization Layer #1
    norm1 = tf.nn.lrn(pool1,
                      depth_radius=4,
                      bias=1.0,
                      alpha=0.001 / 9.0,
                      beta=0.75,
                      name="norm1")

    
    #Convolutional Layer #2 and Pooling Layer #2
    
    conv2 = tf.layers.conv2d(
            inputs=norm1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

    #Normalization Layer #2
    norm2 = tf.nn.lrn(conv2,
                      depth_radius=4,
                      bias=1.0,
                      alpha=0.001 / 9.0,
                      beta=0.75,
                      name="norm2")
    
    pool2 = tf.layers.max_pooling2d(inputs=norm2, pool_size=[5, 5],
            strides=5)


    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 16 * 9 * 64])

    dense = tf.layers.dense(inputs=pool2_flat, units=1024,
            activation=tf.nn.relu,
            use_bias=True,
            name="dropout")

    dropout = tf.layers.Dropout(rate=0.4)

    dropout_output = dropout(dense,
            training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits_layer = tf.layers.dense(inputs=dropout_output, units=8)

    logits = tf.reshape(logits_layer, [-1, 1, 8])

    pred_box1, pred_box2 = tf.split(
            tf.reshape(
                tf.cast(
                    logits, tf.int64)
                , [-1,4,2])
            , [1,1], axis=2)

    

    predictions = {
            "hand1": pred_box1,
            "hand2": pred_box2,
            "output": logits
        }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                predictions=predictions)

    #Calculate Loss (for both TRAIN and EVAL mode)

    loss = tf.losses.absolute_difference(labels, logits)


    #Configure the Training Op (for TRAIN mode)

    if mode == tf.estimator.ModeKeys.TRAIN:

        optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=0.001)

        train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_or_create_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                train_op=train_op)
    
    true_box1, true_box2 = tf.split(
            tf.reshape(
                tf.cast(
                    labels, tf.int64)
                , [-1,4,2])
            , [1,1], axis=2)

      # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "rmse_hand1": tf.metrics.root_mean_squared_error(
          labels=true_box1,
          predictions=pred_box1,
          name="rmse_hand1"
        ),
        "mse_hand1": tf.metrics.mean_squared_error(
             labels=true_box1,
             predictions=pred_box1,
             name="mse_hand1"
        ),
        "mae_hand1": tf.metrics.mean_absolute_error(
             labels=true_box1,
             predictions=pred_box1,
             name="mae_hand1"
        ),
        "rmse_hand2": tf.metrics.root_mean_squared_error(
          labels=true_box2,
          predictions=pred_box2,
          name="rmse_hand2"
        ),
        "mse_hand2": tf.metrics.mean_squared_error(
             labels=true_box2,
             predictions=pred_box2,
             name="mse_hand2"
        ),
        "mae_hand2": tf.metrics.mean_absolute_error(
             labels=true_box2,
             predictions=pred_box2,
             name="mae_hand2"
        )
        }

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                eval_metric_ops=eval_metric_ops)

def rotate_label_90(box, 
                    center=(int(IMAGE_HEIGHT/2),int(IMAGE_WIDTH/2))):
        
    h, w, x, y = tf.split(tf.cast(box, tf.int32), [2,2,2,2])

    points = tf.reshape(tf.concat((y,x), axis=0), [2,2])

    cent = tf.constant(np.array(center).reshape(2,1))

    rot = tf.constant(np.array([[0, -1],
                                [1,  0]]))
   
    adder = tf.add((-1*tf.matmul(rot, cent)),cent)

    canvas_rotation = tf.reverse(cent, axis=[0]) - cent

    step1 = tf.matmul(rot, points)

    step2 = tf.add(adder, step1)
    
    step3 = tf.add(step2,canvas_rotation)

    y, x = tf.split(step3, [1,1], axis=0)

    x = tf.reshape(x, [2])
    y = tf.reshape(y, [2])
    y = tf.subtract(y, w)

    return tf.concat((w, h, x, y), axis=0)

def mirror_label(box, axis=1, shape=(IMAGE_HEIGHT, IMAGE_WIDTH)):
    
    if axis != 1 and axis != 0:
        raise ValueError

    h, w, x, y = tf.split(tf.cast(box, tf.int32), [2,2,2,2])

    points = tf.reshape(tf.concat((y,x), axis=0), [2,2])

    adder = np.array([0, 0]).reshape(2,1)
    index = 1 - axis
    adder[axis] = shape[axis]

    adderTensor = tf.constant(adder)

    rot = np.array([[-1, 0],[0,  1]]) if index == 1 else \
          np.array([[1, 0],[0, -1]])

    rotTensor = tf.constant(rot)
    
    mirror1 = tf.matmul(rotTensor, points)

    mirror2 = tf.add(adder, mirror1)
    
    mirror3 = tf.reverse(mirror2, axis=[0])

    x, y = tf.split(mirror3, [1,1], axis=0)

    x = tf.reshape(x, [2])
    y = tf.reshape(y, [2])

    if axis == 1:
        x = tf.subtract(x, w)
    else:
        y = tf.subtract(y, h)

    return tf.concat((h, w, x, y), axis=0)


def rotate_90(image, label):
    rotated_image = tf.image.rot90(image)
    rotated_label = rotate_label_90(label)

    return rotated_image, rotated_label

def flip_vertical(image, label):
    flipped_image = tf.image.flip_up_down(image)
    flipped_label = mirror_label(label, axis=0)

    return flipped_image, flipped_label

def flip_horizontal(image, label):
    flipped_image = tf.image.flip_left_right(image)
    flipped_label = mirror_label(label)

    return flipped_image, flipped_label

def maybe_transform(transform, image, label):
    maybe = random.choice([True, False])

    if maybe:
        return transform(image, label)
    
    return image, label

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

    ##############################################################
    ################TODO: MaybeRemoveDistortions##################
    #############################################################

    dis_image = decoded_image

    # image_distortions = [partial(tf.image.random_brightness,
    #                              max_delta=63),
    #                      partial(tf.image.random_contrast,
    #                              lower=0.2,
    #                              upper=1.8)
    #                      ]
    # random.shuffle(image_distortions)

    # for dis in image_distortions:
    #     dis_image = dis(image=dis_image)
    
    float_image = tf.image.per_image_standardization(dis_image)

    float_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    #dis_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    return float_image
    
def build_double_dataset(filename, label):
    image_dataset = tf.data.Dataset.from_tensors(filename)
    image_dataset = image_dataset.map(_decode_images)

    label_dataset = tf.data.TFRecordDataset(label)
    label_dataset = label_dataset.map(_decode_labels)

    return tf.data.Dataset.zip((image_dataset, label_dataset))


def _apply_distortions_to_dataset(image, label):
    #Transforms
    #Transformations in a random list since non cummutative

    ###################TODO: COMENT BACK IN######################
    #transforms = [flip_vertical, flip_horizontal, rotate_90]   #
    #random.shuffle(transforms)                                 #
    #                                                           #
    #trans_image = image                                        #
    #trans_label = label                                        #
    #                                                           #
    #for trans in transforms:                                   #
    #    trans_image, trans_label = maybe_transform(trans,      #
    #                                               trans_image,# 
    #                                               trans_label)#
    #############################################################

    dis_image = image

    image_distortions = [partial(tf.image.random_brightness,
                                 max_delta=63),
                         partial(tf.image.random_contrast,
                                 lower=0.2,
                                 upper=1.8)
                         ]
    random.shuffle(image_distortions)

    for dis in image_distortions:
        dis_image = dis(image=dis_image)
    
    #trans_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    float_image = tf.image.per_image_standardization(dis_image)

    float_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    return float_image, trans_label

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
    #dataset = dataset.map(_apply_distortions_to_dataset)
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

