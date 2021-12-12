# USAGE
# python detect_mask_image.py --image images/pic1.jpeg

# import the necessary packages
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from utils import visualize_image
import numpy as np
import argparse
import logging
import cv2
import glob
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detect_mask_images")

def detect_single_mask(image, face_detector, mask_classifier, detect_confidence):
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(640, 640),
                                mean=(104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()

    faces = []
    locs = []
    preds = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > detect_confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = image[startY:endY, startX:endX]
            if face.any():
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                
                faces.append(face)
                locs.append((startX, startY, endX, endY))
    
    # only make a prediction if at least one face was detected
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        with tf.device("/cpu:0"):
            preds = mask_classifier.predict(faces, batch_size=32)

    return (locs, preds)

def detect_mask_images(image_path_list, face_path, mask_path, detect_confidence, visualize=False):
    logger.info("Loading face detection model...")
    prototxt_path = os.path.join(face_path, "deploy.prototxt")
    weights_path = os.path.join(face_path, "res10_300x300_ssd_iter_140000.caffemodel")
    detector = cv2.dnn.readNet(prototxt_path, weights_path)

    logger.info("Loading mask classification model...")
    with tf.device('/cpu:0'):
        classifier = load_model(mask_path)

    res_list = []
    for image_path in image_path_list:
        image = cv2.imread(image_path)
        (locs, preds) = detect_single_mask(image, detector, classifier, detect_confidence)
        if visualize:
            visualize_image(image, locs, preds)
        res_list.append((locs, preds, image_path))
    return res_list


if __name__ == "__main__":
    image_path_list = glob.glob('data/*')
    face_path = 'face_detect/models'
    mask_path = 'mask_classifier/models/mask_detector.model'
    detect_confidence = 0.1
    res_list = detect_mask_images(image_path_list, face_path, mask_path, detect_confidence)
    print(res_list)


