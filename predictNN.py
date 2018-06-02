import cv2
import numpy as np
import tensorflow as tf
import cnn_box


if __name__ == "__main__":

    input_data = np.load("Gloved_Hand_Training_Data.npz")
    
    segmentation = np.load("Gloved_Hand_Labels.npz")

    cnn = tf.estimator.Estimator(model_fn=cnn_box.cnn_model_fn,
            model_dir="box_checkpoints")

    for key, frame in input_data.items():

        expected_frame = frame.copy()

        actual_frame = frame.copy()

        input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x":np.array([actual_frame.astype(np.float32)])},
                shuffle=False,
                batch_size=1)
    
        cnn_output = cnn.predict(input_fn)
        actual_box = next(cnn_output)['answer'].reshape(4, 2).astype(np.int64)

        expected_box = segmentation[key]

        expected_frame = cv2.drawContours(expected_frame,
                [expected_box], 0, (0, 255, 255), 5)

        actual_frame = cv2.drawContours(actual_frame,
                [actual_box], 0, (255,0,255), 5)

        cv2.imshow('expected', expected_frame)
        cv2.imshow('actual', actual_frame)
        
        k = cv2.waitKey(50) & 0xFF
        if k == ord('q'):
            break

    cv2.destroyAllWindows()






