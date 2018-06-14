from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

IMAGE_WIDTH = 320

IMAGE_HEIGHT = 180

CHECKPOINT_DIR = "surgical_lnr_checkpoints"

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
