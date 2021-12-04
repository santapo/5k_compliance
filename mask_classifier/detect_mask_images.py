# USAGE
# python detect_mask_image.py --image images/pic1.jpeg

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import tflite_runtime.interpreter as tflite
from tensorflow.keras.models import load_model
# from retinaface import RetinaFace
# from utils_5k import visualize_image
import numpy as np
import pandas as pd
import argparse
import logging
import cv2
import glob
import os
from natsort import natsorted
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detect_mask_images")

def detect_single_mask(image_path, faceNet):
    locs = []
    faces = []
    # grab the dimensions of the frame and then construct a blob from it
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (640, 640),
                                    (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            try:
                # extract the face ROI
                # startX, startY: top left corner
                # endX, endY: bottom right corner
                face = image[startY:endY, startX:endX]

                # resize it to 160x160
                face = cv2.resize(face, (160, 160))

                # expand array shape from [160, 160, 3] to [1, 160, 160, 3]
                face = np.expand_dims(face, axis=0)

                # add the face and bounding boxes to their respective lists
                faces.append(face)
                locs.append((startY, startX, endY, endX))
            except:
                pass
    return locs, faces


def detect_mask_image(image_path, faceNet, interpreter, input_details, output_details):
    locs, faces = detect_single_mask(image_path, faceNet)
    '''tflite'''
    labels = []
    scores = []

    for face in faces:
        # pre-process image to conform to MobileNetV2
        # input_mean = input_std = float(127.5)
        # input_data = (np.float32(face) - input_mean) / input_std

        # set our input tensor to our face image
        interpreter.set_tensor(input_details[0]['index'], np.float32(face))

        # perform classification
        interpreter.invoke()

        # get our output results tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        result = np.squeeze(output_data)

        # get label from the result.
        # the class with the higher confidence is the label.
        (mask, withoutMask) = result
        label = "Mask" if mask > withoutMask else "No Mask"

        # get the highest confidence as the label's score
        score = np.max(result)

        labels.append(label)
        scores.append(score)
    return (labels, scores)

def detect_all_images(image_path_list, face_path):
    interpreter = tflite.Interpreter(
        # model_path=os.path.expanduser("/home/pi/CollegeProject/models/MaskDetector/model.tflite"))
        model_path=os.path.expanduser("model_quant.tflite"))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    logger.info("Loading face detection model...")
    prototxt_path = os.path.join(face_path, "deploy.prototxt")
    weights_path = os.path.join(face_path, "res10_300x300_ssd_iter_140000.caffemodel")
    faceNet = cv2.dnn.readNet(prototxt_path, weights_path)
    
    image_labels = []
    for image_path in tqdm(image_path_list, desc='detecting mask'):
        label, scores = detect_mask_image(image_path, faceNet, interpreter, input_details, output_details)
        image_label = 0 if "No Mask" in label else 1
        image_labels.append((image_path, image_label))
    
    return image_labels
if __name__ == "__main__":
    image_path_list = glob.glob('data/public_test/images/*.jpg')
    all_res_mask = detect_all_images(image_path_list, 'face_model/')
    with open('mask_detect_result.txt', 'w') as f:
        f.write(str(all_res_mask))
    df = pd.read_csv('distancing.csv')
    res = []
    for path, mask_label in all_res_mask:
        path = os.path.basename(path)
        final_label = int(df[df['fname'] == path]['distancing'])*mask_label
        res.append((path, final_label))
    from utils_5k import merge_result_to_meta
    meta_path = "/home/michael/Michael/zaloai_challenge/5k_compliance/data/public_test/public_test_meta.csv"
    final_df = merge_result_to_meta(meta_path, res)
    final_df.to_csv("submit_public_test_person.csv")

