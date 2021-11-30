# USAGE
# python detect_mask_image.py --image images/pic1.jpeg

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from retinaface import RetinaFace
from utils_5k import visualize_image
import numpy as np
import argparse
import logging
import cv2
import glob
import os
from natsort import natsorted
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detect_mask_images")

def detect_single_mask(image_path, locs, mask_classifier):
    image = cv2.imread(image_path)
    faces = []
    for loc in locs:
        startX, startY, endX, endY = loc
        startX = int(startX); startY = int(startY); endX = int(endX); endY = int(endY)
        # import ipdb; ipdb.set_trace()
        face = image[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        faces.append(face)

    # only make a prediction if at least one face was detected
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = mask_classifier.predict(faces, batch_size=32)

    return (locs, preds)

def detect_mask_images(args, image_path_list, all_locs, visualize=False):
    logger.info("Loading mask classification model...")
    classifier = load_model(args.mask)
    image_path_list = natsorted(image_path_list)
    res_list = []
    for image_path, locs in tqdm(zip(image_path_list, all_locs), total=len(image_path_list), desc="detecting mask"):
        (_locs, preds) = detect_single_mask(image_path, locs, classifier)
        if visualize:
            image = cv2.imread(image_path)
            visualize_image(image, _locs, preds)
        res_list.append((_locs, preds, image_path))
    return res_list


if __name__ == "__main__":
    from parser_5k import set_parser
    args = set_parser()
    image_path_list = glob.glob(f'{args.images_dir}/*.jpg')
    res = detect_mask_images(args, image_path_list)
    print(res)

