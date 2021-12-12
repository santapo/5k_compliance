import argparse
import glob
import os

import numpy as np
import pandas as pd

from utils import merge_result_to_meta
from tools.distance_calculate import calculate_distance
from mask_classifier.detect_mask_images import detect_mask_images

def mask_compliance(preds):
    is_mask = preds[:, 0] > preds[:, 1]
    if is_mask.all():
        return True
    return False

def distancing_compliance(dist, dist_threshold):
    if dist < dist_threshold:
        return False
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-i", "--images_dir", type=str, default=None,
    #                     help="Path to images directory")
    parser.add_argument('dir', nargs='?', default=os.getcwd())
    parser.add_argument("-f", "--face", type=str, default="face_detect/models",
                        help="Path to face detection model directory")
    parser.add_argument("-m", "--mask", type=str, default="mask_classifier/mask_detector.model",
                        help="Path to mask classification model directory")
    parser.add_argument("-dc", "--detect_confidence", type=float, default=0.5,
                        help="Minimum probability to filer weak detections")
    parser.add_argument("-cp", "--load_ckpt", type=str, default="./ckpt/res50.pth",
                        help="Depth model's checkpoint")
    parser.add_argument("-b", "--backbone", type=str, default="resnet50",
                        help="Depth model's backbone")
    args = vars(parser.parse_args())

    image_path_list = glob.glob(os.path.join(args["dir"], "*"))
    distance_calc_list = calculate_distance(args["load_ckpt"], args["backbone"], image_path_list, 
                            args["face"], args["detect_confidence"])
    mask_detection_list = detect_mask_images(image_path_list, args["face"], args["mask"], args["detect_confidence"], visualize=False)

    result_list = []
    idx = 1
    from tqdm import tqdm
    for dist, res_mask in tqdm(zip(distance_calc_list, mask_detection_list), total=len(image_path_list), desc="5k compliance on public test"):
        # locs: List of tuple, preds numpy array (N, 2)
        locs, preds, image_path = res_mask
        image_name = os.path.basename(image_path)
        # Continue if prediction is empty
        is_distance_compliance = distancing_compliance(dist, 0.12)
        is_mask_compliance = mask_compliance(preds)
        is_5k_compliance = int(is_mask_compliance and is_distance_compliance)
        result_list.append((idx, image_name, is_5k_compliance))
        idx += 1

    final_df = pd.DataFrame(result_list, columns=["image_id", "fname", "5K"])
    final_df.to_csv("submission.csv")



