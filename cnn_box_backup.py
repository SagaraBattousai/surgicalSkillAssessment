from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import argparse

import cnn_box_model


TO_FLOAT32 = lambda a: a.astype(np.float32)

def cmdline_predict(numpy_file, indecies=None,
                    names=None, slices=None):
  
  input_data = np.load(numpy_file)

  data = []

  if names != None:
    data = [input_data[name] for name in names]
    return predict(*data)
    

  input_data_list = list(dict(all_data).values())

  
  if indecies != None:
    data = [input_data_list[index] for index in indecies]

  elif slices != None:
    if len(slices) == 1:
      data = input_data_list[0:slices[0]]

    elif len(slices) > 1:
      data = input_data_list[slices[0]:slices[1]]

  else:
      data = input_data_list

  return predict(*data)

def predict(*input_data):

  cnn = tf.estimator.Estimator(model_fn=cnn_box_model.cnn_model_fn,
      model_dir=cnn_box_model.CHECKPOINT_DIR)

  data = np.array(list(map(TO_FLOAT32, input_data)))

  input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": data},
        shuffle=False,
        batch_size=1)

  return cnn.predict(input_fn)
  



def main(unsused_argv):

    all_data = np.load("Gloved_Hand_Training_Data.npz")
    all_labels = np.load("Gloved_Hand_Labels.npz")
    
    all_data_list = np.array(list(dict(all_data).values()), dtype=np.float32)
    all_labels_list = np.array(list(dict(all_labels).values()), dtype=np.float32).reshape(-1, 8)

    train_data = all_data_list[0:2000]
    train_labels = all_labels_list[0:2000]#np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = all_data_list[-810:2810]
    eval_labels = all_labels_list[-810:2810]

    #Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
            model_fn=cnn_box_model.cnn_model_fn, 
            model_dir="box_checkpoints")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=100,
            num_epochs=None,
            shuffle=True)

    mnist_classifier.train(
            input_fn=train_input_fn,
            steps=20000,
            hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)


def train():
  tf.logging.set_verbosity(tf.logging.INFO)

  tf.app.run()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.set_defaults(func=train)


    subparsers = parser.add_subparsers(dest="subparser_name")


    trainArgs = subparsers.add_parser('train')

    trainArgs.set_defaults(func=train)



    predictArgs = subparsers.add_parser('predict')

    predictArgs.add_argument('numpy_file')

    predictGroup = predictArgs.add_mutually_exclusive_group()

    predictGroup.add_argument('-i', '--indecies', nargs='+')

    predictGroup.add_argument('-n', '--names', nargs='+')

    predictGroup.add_argument('-s', '--slices', nargs='+', type=int)

    predictArgs.set_defaults(func=cmdline_predict)



    args = parser.parse_args()


    if args.subparser_name == "predict":
      args.func(args.numpy_file, args.indecies, 
                args.names, args.slices)

    else:
      args.func()
























