from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

IMAGE_WIDTH = 320

IMAGE_HEIGHT = 180

INITIAL_LEARNING_RATE = 0.1

LEARNING_RATE_DECAY = 0.1

def cnn_model_fn(features, labels, mode):
    #Input layer
    input_layer = tf.reshape(tf.cast(features, tf.float32), [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    ##MAYBE ADD VARIABLE SCOPE
    
    #Convolutional Layer #1
    conv1 = tf.layers.Conv2D(filters=64,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu,
                             name="conv1")

    conv1_out = conv1(input_layer)

    #Pooling Layer #1
    max_pool_quarter = tf.layers.MaxPooling2D(pool_size=[4, 4],
                                              strides=4,
                                              name="quarter_pool")

    pool1_out = max_pool_quarter(conv1_out)

    #Normalization Layer #1
    # norm1 = tf.nn.lrn(pool1_out, 
    #                   depth_radius=4,
    #                   bias=1.0,
    #                   alpha=0.001 / 9.0,
    #                   beta=0.75,
    #                   name="norm1")

    #Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.Conv2D(filters=64,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu,
                             name="conv2")

    conv2_out = conv2(pool1_out)
    
    #Normalization Layer #2
    # norm2 = tf.nn.lrn(conv2_out, 
    #                   depth_radius=4,
    #                   bias=1.0,
    #                   alpha=0.001 / 9.0,
    #                   beta=0.75,
    #                   name="norm2")


    #Pooling Layer #2
    max_pool_fifth = tf.layers.MaxPooling2D(pool_size=[5, 5],
                                             strides=5,
                                             name="fifth_pool")

    pool2_out = max_pool_fifth(conv2_out)

    # Dense Layer
    flattend_out = tf.reshape(pool2_out, [-1, 16 * 9 * 64])

    dense = tf.layers.dense(inputs=flattend_out, 
                            units=1024,
                            activation=tf.nn.relu,
                            name="dense")

    #Moved from 0.4 to 0.2 (because of less data???) Back to 0.4
    dropout = tf.layers.dropout(inputs=dense, rate=0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN,
        name="dropout")


    # final Layer
    dense2 = tf.layers.Dense(units=8,
                                   name="output_layer")


    output_layer = dense2(dropout)
    
    output = tf.reshape(output_layer, [-1, 1, 8])

    pred_box1, pred_box2 = tf.split(
            tf.reshape(
                tf.cast(
                    output, tf.int64)
                , [-1,4,2])
            , [1,1], axis=2)

    

    predictions = {
            "hand1": pred_box1,
            "hand2": pred_box2,
            "output": output
        }

    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode,
          predictions=predictions)

      #Calculate Loss (for both TRAIN and EVAL mode)
    true_box1, true_box2 = tf.split(tf.reshape(labels, [-1,4,2]), [1,1], axis=2)

    loss = tf.losses.absolute_difference(labels, output)

    #Configure the Training Op (for TRAIN mode)

    if mode == tf.estimator.ModeKeys.TRAIN:
        
        batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
        decay_steps = int(batches_per_epoch * EPOCH_PER_DECAY)


        learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    tf.train.get_or_create_global_step(),
                                    decay_steps,
                                    LEARNING_RATE_DECAY,
                                    staircase=True)

        
        optimizer = tf.train.AdamOptimizer(
          learning_rate=learning_rate)

        train_op = optimizer.minimize(
          loss=loss,
          global_step=tf.train.get_or_create_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
          train_op=train_op)

      # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        #    "l2_hand1": tf.nn.l2_loss((true_box1, pred_box1), name="p"),
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
