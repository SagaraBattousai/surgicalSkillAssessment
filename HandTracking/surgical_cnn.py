from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import argparse
import cv2
import cvTools
import os

from matchers import Boundary
from functools import partial

import surgical_cnn_input as sci

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
def _parse_image(filename, standardise=False):
    image_string = tf.read_file(filename)#image_filename)

    decoded_image =tf.image.decode_image(image_string, channels=1) 

    if standardise:

        float_image = tf.image.per_image_standardization(decoded_image)

        float_image.set_shape([sci.IMAGE_HEIGHT, sci.IMAGE_WIDTH, 1])

    else:
        decode_image = tf.cast(decoded_image, tf.float32)
        decoded_image.set_shape([sci.IMAGE_HEIGHT, sci.IMAGE_WIDTH, 1])
        
        float_image = decoded_image

    return float_image

def predictionData(filenames, standardise=False):

    files = tf.constant(filenames)
    dataset = tf.data.Dataset.from_tensor_slices((files))

    parse_fn = partial(_parse_image, standardise=standardise)

    dataset = dataset.map(parse_fn)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(1)

    return dataset

def predict(input_fn):

    cnn = tf.estimator.Estimator(model_fn=sci.cnn_model_fn,
      model_dir=sci.PREDICTION_CHECKPOINT_DIR)

    predictions = cnn.predict(input_fn=input_fn)

    return predictions

def overlayBox(image, prediction, copy=False):


    hand1 = prediction['hand1']

    hand2 = prediction['hand2']
   
    h1, w1, x1, y1 = hand1

    h2, w2, x2, y2 = hand2

    box1 = Boundary.fromRect(x1, y1, w1, h1)
        
    box2 = Boundary.fromRect(x2, y2, w2, h2)
    
    img = image.copy() if copy else image

    box1.drawBoundary(img)

    box2.drawBoundary(img)
        
    return img

def cmdline_predict(image_filenames, standardise=False):

    if not isinstance(image_filenames, (tuple, list)):
        image_filenames = [image_filenames]
    
    input_fn = partial(predictionData, image_filenames, standardise=standardise)
    
    predictions1 = predict(input_fn)

    outputs = []

    for i, x in enumerate(predictions1):
    
        img = cv2.imread(image_filenames[i]).astype(np.uint8)
        #print(x['output'])
        
        outputs.append((overlayBox(img, x), x))

    for x in outputs:
        cvTools.displayImages(x[0])
        print(x[1])

##def playSeries(series, f=lambda x : x, speed=30):

def playAsSeries(image_filenames, standardise, speed=40):

    if not isinstance(image_filenames, (tuple, list)):
        image_filenames = [image_filenames]
    
    input_fn = partial(predictionData, image_filenames, standardise=standardise)
    
    predictions1 = predict(input_fn)

    outputs = []

    for i, x in enumerate(predictions1):
    
        img = cv2.imread(image_filenames[i]).astype(np.uint8)
        #print(x['output'])
        
        outputs.append(overlayBox(img, x))

    cvTools.playSeries(outputs, speed=speed)

def runLive(capture=2):

    cap = cv2.VideoCapture(capture)

    widthSet = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

    heightSet = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        cap.open()

    ret, frame = cap.read()

    try:
        while(ret):

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            frame = cv2.resize(frame, None, fx=1/4, fy=1/4, interpolation=cv2.INTER_CUBIC)

            input_fn = lambda : tf.data.Dataset.from_tensors(frame)

            prediction = predict(input_fn)

            pred = next(prediction)

            frame = overlayBox(frame, pred)

            cv2.imshow('Capture', frame)
    
            k = cv2.waitKey(25) & 0xFF
            if k == ord('q'):
                break
        
            ret, frame = cap.read()

    finally:
        cv2.destroyAllWindows()
        cap.release()


        tensor = tf.convert_to_tensor(frame)
        # dataset = tf.data.Dataset.from_tensors(tensor)
        # dataset = dataset.repeat(1)#Needed?????
        k = tf.estimator.inputs.numpy_input_fn(frame.astype(np.uint8), shuffle=False)

        prediction = predict(k)#lambda : k)

        print(prediction)

        pred = next(prediction)

        return overlayBox(frame, pred)

    cvTools.record_while(predictLiveImage)




  
def main(unsused_argv):

    #Checkpoint config
    checkpointing_config = tf.estimator.RunConfig(
                                        save_checkpoints_steps=1000,
                                        keep_checkpoint_max=10)
    #Create the Estimator
    hand_tracking_regressor = tf.estimator.Estimator(
            model_fn=sci.cnn_model_fn, 
            model_dir=sci.CHECKPOINT_DIR,
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
            input_fn=sci.get_train_data)#,
            #hooks=[logging_hook])

    eval_results = hand_tracking_regressor.evaluate(input_fn=sci.get_eval_data)


def train():
  tf.logging.set_verbosity(tf.logging.INFO)

  tf.app.run()

def eval():
    hand_tracking_regressor = tf.estimator.Estimator(
            model_fn=sci.cnn_model_fn, 
            model_dir=sci.CHECKPOINT_DIR)
    
    eval_results = hand_tracking_regressor.evaluate(input_fn=sci.get_eval_data)

    print(eval_results)

def getAll(baseDir):
    image_names = os.listdir(baseDir)[:2000]

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
    
    predictArgs.add_argument('-s', '--standardise', action='store_true')

    predictArgs.set_defaults(func=cmdline_predict)

    
    predict_allArgs = subparsers.add_parser('all')

    predict_allArgs.add_argument('baseDir', nargs='?', default=DATA_PREDICT_LOCATION, type=getAll)
    
    predict_allArgs.add_argument('-s', '--standardise', action='store_true')
    
    predict_allArgs.add_argument('-v', '--video', action='store_true')

    predict_allArgs.set_defaults(func=cmdline_predict)



    args = parser.parse_args()


    if args.subparser_name == "predict":
        args.func(args.images, args.standardise)
    
    elif args.subparser_name == "all":
        if args.video:
            playAsSeries(args.baseDir, args.standardise)
        else:
            args.func(args.baseDir, args.standardise)

    else:
      args.func()
























