from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import argparse
import cv2
import cvTools
import os
import importlib
import threading

import tip_cnn_input as sci

from matchers import Boundary
from functools import partial

model = None

TO_FLOAT32 = lambda a: a.astype(np.float32)

REQUIRED_ASPECT_RATION = "16:9"

IMAGE_TO_PREDICT = 'data' + os.sep + 'images' + os.sep + 'frame_3854.jpg'

DATA_PREDICT_LOCATION = 'data' + os.sep + 'images'

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

class BadInputModule(Exception):
    
    def __init__(self, className):
        self.className = className

    def message(self):
        return "Input Module {} Error: Maybe The Model Is Incorrect Or The\
                CheckPoints Are Bad?\nIs The InputFunctionCorrect?".format(
                self.className)


class NoInputModule(BadInputModule):

    def message(self):
        return "No Input Module supplied: Perhaps The Model Is Incorrect Or\
                it Failed To Load?"

def predictionData(filenames):

    files = tf.constant(filenames)
    dataset = tf.data.Dataset.from_tensor_slices((files))
    
    if model is None:
        raise NoInputModule

    dataset = dataset.map(sci._decode_images)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(1)

    return dataset

def _decode_live(img):
    float_image = tf.image.per_image_standardization(img)
    float_image.set_shape([sci.IMAGE_HEIGHT, sci.IMAGE_WIDTH, 1])

    return float_image

def decode_live_images(images):

    imgs = tf.constant(images)
    dataset = tf.data.Dataset.from_tensor_slices((imgs))
    
    if model is None:
        raise NoInputModule

    dataset = dataset.map(_decode_live)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(1)

    return dataset



def predict(input_fn):

    cnn = tf.estimator.Estimator(model_fn=model.cnn_model_fn,
      model_dir=model.CHECKPOINT_DIR)

    predictions = cnn.predict(input_fn=input_fn)

    return predictions

def predictFromFilenames(image_filenames):

    if not isinstance(image_filenames, (tuple, list)):
        image_filenames = [image_filenames]

    input_fn = partial(predictionData, image_filenames)
    
    return predict(input_fn)

def overlayBox(image, prediction, copy=False):


    hand1 = prediction['hand1']

    hand2 = prediction['hand2']
   
    h1, w1, x1, y1 = hand1

    h2, w2, x2, y2 = hand2

    if x1 < x2:
        box1 = Boundary.fromRect(x1, y1, w1, h1)
        box2 = Boundary.fromRect(x2, y2, w2, h2)
    else:
        box2 = Boundary.fromRect(x1, y1, w1, h1)
        box1 = Boundary.fromRect(x2, y2, w2, h2)

    img = image.copy() if copy else image

    box1.drawBoundary(img, colour=(255,255,0), width=2)

    box2.drawBoundary(img, colour=(255,0,255), width=2)
        
    return img

###Must NOT be concureently run with other calls to tensorflow (I think) makes sure to run after other calls and joint before later calls
def savePredictedSequence(image_filenames):
    
    
    #Justifiable Duplication based off unliklyhood of save feature
    
    if not isinstance(image_filenames, (tuple, list)):
        image_filenames = [image_filenames]

    predictions = predictFromFilenames(image_filenames)

    imageFromIndex = lambda index:\
                cv2.imread(image_filenames[index]).astype(np.uint8)

    outputs =   (
                    (overlayBox(imageFromIndex(i), x), x)

                    for i, x in enumerate(predictions)
                )

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('predictionOutput.avi', fourcc, 20.0, (320, 180))

    try:
        for x in outputs:
            out.write(x[0])

    finally:
        out.release()

def cmdline_predict(image_filenames, save, asVideo=False, speed=40):
    
    if not isinstance(image_filenames, (tuple, list)):
        image_filenames = [image_filenames]

    predictions = predictFromFilenames(image_filenames)

    imageFromIndex = lambda index:\
                cv2.imread(image_filenames[index]).astype(np.uint8)

    outputs =   (
                    (overlayBox(imageFromIndex(i), x), x)

                    for i, x in enumerate(predictions)
                )

    #Called after Tensorflow calls but before displaying to caller so
    #Thread can run whilst displaying.
    #Ensure Join before exiting function/catch keyboard interupt to
    #continue saving!
    if save:
        saveThread = threading.Thread(target=savePredictedSequence,
                kwargs={'image_filenames':image_filenames})
        
        saveThread.start()

    try:
        if asVideo:
            outputImages = list(zip(*outputs))[0]
            cvTools.playSeries(outputImages, speed=speed)
        else:
            for x in outputs:
                cvTools.displayImages(x[0])
                print(x[1])

    except KeyboardInterrupt:
        pass

    if save:
        saveThread.join()
    

def cmdline_all(baseDir, start, end, asVideo, speed, save):
    
    if end is None:
        image_names = os.listdir(baseDir)[start:]
    else:
        image_names = os.listdir(baseDir)[start:end]

    images = list(map(lambda a: baseDir + os.sep + a, image_names))

    cmdline_predict(images, save, asVideo, speed)


def runLive(capture=2):

    cap = cv2.VideoCapture(capture)

    widthSet = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

    heightSet = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    if not cap.isOpened():
        cap.open()

    ret, frame = cap.read()

    try:
        while(ret):

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            frame = cv2.resize(frame, None, fx=1/2, fy=1/2, interpolation=cv2.INTER_CUBIC)

            input_fn = partial(decode_live_images, [frame])

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


    ##What was this!!!
        # tensor = tf.convert_to_tensor(frame)
        # # dataset = tf.data.Dataset.from_tensors(tensor)
        # # dataset = dataset.repeat(1)#Needed?????
        # k = tf.estimator.inputs.numpy_input_fn(frame.astype(np.uint8), shuffle=False)

        # prediction = predict(k)#lambda : k)

        # print(prediction)

        # pred = next(prediction)

        # return overlayBox(frame, pred)

    # cvTools.record_while(predictLiveImage)

  
def main(unsused_argv):

    #Checkpoint config
    checkpointing_config = tf.estimator.RunConfig(
                                        save_checkpoints_steps=1000,
                                        keep_checkpoint_max=10)
    #Create the Estimator
    hand_tracking_regressor = tf.estimator.Estimator(
            model_fn=model.cnn_model_fn, 
            model_dir=model.CHECKPOINT_DIR,
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
            model_fn=model.cnn_model_fn, 
            model_dir=sci.CHECKPOINT_DIR)
    
    eval_results = hand_tracking_regressor.evaluate(input_fn=sci.get_eval_data)

    print(eval_results)


def import_model(model_module):
    
    try:
        global model
        model = importlib.import_module(model_module)
    except Exception as e:
        raise BadInputModule(model_module)
        raise e

if __name__ == "__main__":


    ###Base Parser###
    parser = argparse.ArgumentParser()

    parser.set_defaults(func=train)

    subparsers = parser.add_subparsers(dest="subparser_name")


    ##Parent Parser###
    parent_parser = argparse.ArgumentParser(add_help=False)

    parent_parser.add_argument('-m', '--model', nargs='?', default='tip_cnn_model',
                        type=import_model)


    ###Training Parser###
    trainArgs = subparsers.add_parser('train', parents=[parent_parser])

    trainArgs.set_defaults(func=train)
    

    ###Eval Parser###
    evalArgs = subparsers.add_parser('eval', parents=[parent_parser])

    evalArgs.set_defaults(func=eval)


    ###Prediction Parent Parser###
    prediction_parent_parser = argparse.ArgumentParser(add_help=False)

    prediction_parent_parser.add_argument('-sa', '--save', action='store_true')

    
    ###Predict Parser###
    predictArgs = subparsers.add_parser('predict', 
            parents=[parent_parser, prediction_parent_parser])

    predictArgs.add_argument('images', nargs='*', default=IMAGE_TO_PREDICT)
    
    predictArgs.set_defaults(func=cmdline_predict)

   
    ###Predict All Parser
    predict_allArgs = subparsers.add_parser('all', 
            parents=[parent_parser,prediction_parent_parser])

    predict_allArgs.add_argument('baseDir', nargs='?', default=DATA_PREDICT_LOCATION)

    predict_allArgs.add_argument('-s', '--start', default=0, type=int)

    predict_allArgs.add_argument('-e', '--end', default=None, type=int)
    
    predict_allArgs.add_argument('-v', '--video', action='store_true')
    
    predict_allArgs.add_argument('-sp', '--speed', default=40)

    predict_allArgs.set_defaults(func=cmdline_all)

    
    ###Collect args###
    args = parser.parse_args()

    ##Run function for given command###
    if args.subparser_name == "predict":
        args.func(args.images, args.save)
    
    elif args.subparser_name == "all":
        args.func(args.baseDir, args.start, args.end, args.video,
                args.speed, args.save)

    else:
      args.func()
























