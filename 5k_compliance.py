import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch

from utils_5k import merge_result_to_meta
from tools.distance_calculate import calculate_distance
from mask_classifier.detect_mask_images import detect_mask_images
from head_detect import detect

def mask_compliance(preds, locs, image_path):
    try:
        is_mask = preds[:, 0] > preds[:, 1]
    except:
        print("List tuple error in image: {}, length of locs:{}".format(image_path, len(locs)))
        return True
    if is_mask.all():
        return True
    return False

def distancing_compliance(dist, dist_threshold):
    if dist < dist_threshold:
        return False
    return True

if __name__ == "__main__":
    from parser_5k import set_parser
    args = set_parser()
    with torch.no_grad():
        all_locs = detect(args)
    image_path_list = glob.glob(os.path.join(args.images_dir, "*"))
    distance_calc_list = calculate_distance(args, all_locs, image_path_list)
    mask_detection_list = detect_mask_images(args, image_path_list, all_locs, visualize=False)

    result_list = []
    from tqdm import tqdm
    for dist, mask_pred in tqdm(zip(distance_calc_list, mask_detection_list), total=len(image_path_list), desc="5k compliance on public test"):
        # locs: List of tuple, preds numpy array (N, 2)
        locs, preds, image_path = mask_pred
        # import ipdb; ipdb.set_trace()
        image_name = os.path.basename(image_path)
        # Continue if prediction is empty
        is_distance_compliance = distancing_compliance(dist, 0.08)
        is_mask_compliance = mask_compliance(preds, locs, image_path)
        is_5k_compliance = int(is_mask_compliance and is_distance_compliance)
        result_list.append((image_name, is_5k_compliance))

    meta_path = "/home/michael/Michael/zaloai_challenge/5k_compliance/data/public_test/public_test_meta.csv"
    final_df = merge_result_to_meta(meta_path, result_list)
    final_df.to_csv("submit_public_test_person.csv")



