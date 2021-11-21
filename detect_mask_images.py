# USAGE
# python detect_mask_image.py --image images/pic1.jpeg

# import the necessary packages
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

def detect_single_image(image, face_detector, mask_classifier, detect_confidence):
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300),
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
        preds = mask_classifier.predict(faces, batch_size=32)

    return (locs, preds)

def detect_mask_images(image_path_list, face_path, mask_path, detect_confidence, visualize=False):
    logger.info("Loading face detection model...")
    prototxt_path = os.path.join(face_path, "deploy.prototxt")
    weights_path = os.path.join(face_path, "res10_300x300_ssd_iter_140000.caffemodel")
    detector = cv2.dnn.readNet(prototxt_path, weights_path)

    logger.info("Loading mask classification model...")
    classifier = load_model(mask_path)

    res_list = []
    for image_path in image_path_list:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (400, 400))
        (locs, preds) = detect_single_image(image, detector, classifier, detect_confidence)
        if visualize:
            visualize_image(image, locs, preds)
        res_list.append((locs, preds, image_path))
    return res_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images_dir", type=str, default=None,
                        help="Path to images directory")
    parser.add_argument("-f", "--face", type=str, default="face_detector/model",
                        help="Path to face detection model directory")
    parser.add_argument("-m", "--mask", type=str, default="mask_classifier/mask_detector.model",
                        help="Path to mask classification model directory")
    parser.add_argument("-dc", "--detect_confidence", type=float, default=0.5,
                        help="Minimum probability to filer weak detections")
    args = vars(parser.parse_args())

    logger.info("Loading input images...")
    image_path_list = glob.glob(os.path.join(args["images_dir"], "*"))
    res_list = detect_mask_images(image_path_list, args["face"], args["mask"], args["detect_confidence"], visualize=True)

    




