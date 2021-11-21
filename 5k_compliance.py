import argparse
import glob
import os

import numpy as np
import pandas as pd

from utils import merge_result_to_meta
from detect_mask_images import detect_mask_images



def euclidean_distance(point_one, point_two):
    return ((point_one[0] - point_two[0]) ** 2 +
            (point_one[1] - point_two[1]) ** 2) ** 0.5

def pairwise_euclidean_distance(points):
    x = np.array([pt[0] for pt in points])
    y = np.array([pt[1] for pt in points])
    dist_matrix = np.sqrt(np.square(x - x.reshape(-1,1)) + np.square(y - y.reshape(-1,1)))
    return dist_matrix

def convert_to_centerpoint(locs):
    centers = []
    for loc in locs:
        (startX, startY, endX, endY) = loc
        centerX = int((startX + endX) / 2)
        centerY = int((startY + endY) / 2)
        centers.append((centerX, centerY))
    return centers

def distancing_compliance(locs: list, dist_threshold):
    centers = convert_to_centerpoint(locs)
    dist_matrix = pairwise_euclidean_distance(centers)
    if (dist_matrix < dist_threshold).any():
        return False
    return True

def mask_compliance(preds):
    is_mask = preds[:, 0] > preds[:, 1]
    if is_mask.all():
        return True
    return False


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

    image_path_list = glob.glob(os.path.join(args["images_dir"], "*"))
    detection_list = detect_mask_images(image_path_list, args["face"], args["mask"], args["detect_confidence"], visualize=False)

    result_list = []
    for locs, preds, image_path in detection_list:
        # locs: List of tuple, preds numpy array (N, 2)
        image_name = os.path.basename(image_path)
        # Continue if prediction is empty
        if len(locs) == 0:
            is_5k_compliance = 1
            result_list.append((image_name, is_5k_compliance))
            continue
        is_mask_compliance = mask_compliance(preds)
        is_distancing_compliance = distancing_compliance(locs, dist_threshold=30)
        is_5k_compliance = int(is_mask_compliance and is_distancing_compliance)
        result_list.append((image_name, is_5k_compliance))

    meta_path = "/home/santapo/OnlineLab/5k_compliance_zalo/5k_compliance/data/public_test/public_test_meta.csv"
    final_df = merge_result_to_meta(meta_path, result_list)
    final_df.to_csv("submit_public_test.csv")    



