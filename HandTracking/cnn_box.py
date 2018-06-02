from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import argparse
import cv2
import cvTools
import os

from matchers import Boundary
from functools import partial

import cnn_box_input as cbi

TO_FLOAT32 = lambda a: a.astype(np.float32)

REQUIRED_ASPECT_RATION = "16:9"

IMAGE_TO_PREDICT = 'data\\images\\frame_3854.jpg'

DATA_PREDICT_LOCATION = 'data\\images'

class InputDataError(Exception):
    pass

class WrongAspectRatio(InputDataError):
    
    CORRECT_ASPECT_RATIO = REQUIRED_ASPECT_RATION
    
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def message(self):
        return "Required Aspect Ratios is {}, but {}:{} was given".format(
                CORRECT_ASPECT_RATIO,
                width,
                height
                )

def _parse_image(filename):
    image_string = tf.read_file(filename)#image_filename)

    return tf.image.decode_image(image_string, channels=1)

def predictionData(filenames):

    files = tf.constant(filenames)
    dataset = tf.data.Dataset.from_tensor_slices((files))
    dataset = dataset.map(_parse_image)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(1)

    return dataset

def cmdline_predict(image_filename):
    
    cnn = tf.estimator.Estimator(model_fn=cbi.cnn_model_fn,
      model_dir=cbi.PREDICTION_CHECKPOINT_DIR)
  
    input_fn = partial(predictionData, image_filename)

    predictions = cnn.predict(input_fn=input_fn)

    outputs = []

    for i, x in enumerate(predictions):
        hand1 = x['hand1']

        hand2 = x['hand2']

        img = cv2.imread(image_filename[i]).astype(np.uint8)
        
        h1, w1, x1, y1 = hand1

        h2, w2, x2, y2 = hand2

        box1 = Boundary.fromRect(x1, y1, w1, h1)
        box2 = Boundary.fromRect(x2, y2, w2, h2)
    
        box1.drawBoundary(img)
        
        outputs.append(box2.drawBoundary(img))

    for x in outputs:
        cvTools.displayImages(x)


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

    #Checkpoint config
    checkpointing_config = tf.estimator.RunConfig(
                                        save_checkpoints_steps=1000,
                                        keep_checkpoint_max=10)
    #Create the Estimator
    hand_tracking_regressor = tf.estimator.Estimator(
            model_fn=cbi.cnn_model_fn, 
            model_dir=cbi.CHECKPOINT_DIR,
            config=checkpointing_config)

    #Set up logging for predictions
    #tensors_to_log = {
    #                    "me":"p"
    #                     "rmse_hand1": "rmse_hand1",
    #                     "mse_hand1": "mse_hand1", 
    #                     "mae_hand1": "mae_hand1",
    #                     "rmse_hand2": "rmse_hand2",
    #                     "mse_hand2": "mse_hand2",
    #                     "mae_hand2": "mae_hand2"
    #                 }
    
    # logging_hook = tf.train.LoggingTensorHook(
    #         tensors=tensors_to_log, every_n_iter=50)



    hand_tracking_regressor.train(
            input_fn=cbi.get_train_data)#,
            #hooks=[logging_hook])

    eval_results = hand_tracking_regressor.evaluate(input_fn=cbi.get_eval_data)


def train():
  tf.logging.set_verbosity(tf.logging.INFO)

  tf.app.run()

def eval():
    hand_tracking_regressor = tf.estimator.Estimator(
            model_fn=cbi.cnn_model_fn, 
            model_dir=cbi.CHECKPOINT_DIR)
    
    eval_results = hand_tracking_regressor.evaluate(input_fn=cbi.get_eval_data)

    print(eval_results)

def getAll(baseDir):
    image_names = os.listdir(baseDir)[:500]

    return list(map(lambda a: baseDir + "\\" + a, image_names))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.set_defaults(func=train)


    subparsers = parser.add_subparsers(dest="subparser_name")


    trainArgs = subparsers.add_parser('train')

    #TODO: CHANGE 
    trainArgs.set_defaults(func=train)

    evalArgs = subparsers.add_parser('eval')

    evalArgs.set_defaults(func=eval)

    
    predictArgs = subparsers.add_parser('predict')

    predictArgs.add_argument('images', nargs='+')

    predictArgs.set_defaults(func=cmdline_predict)

    
    predict_allArgs = subparsers.add_parser('all')

    predict_allArgs.add_argument('baseDir', nargs='?', default=DATA_PREDICT_LOCATION, type=getAll)

    predict_allArgs.set_defaults(func=cmdline_predict)



    args = parser.parse_args()


    if args.subparser_name == "predict":
        args.func(args.images)
    
    elif args.subparser_name == "all":
        args.func(args.baseDir)

    else:
      args.func()
























