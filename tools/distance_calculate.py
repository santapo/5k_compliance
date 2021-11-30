import cv2
import numpy as np
import torch
from tools.test_depth import predict_depths
from tqdm import tqdm
from natsort import natsorted

from head_detect import detect
from utils_5k import list_of_txt_to_locs

def calculate_centroid_coordination(image, locs):
    (h, w) = image.shape[:2]
    centroids = []
    if len(locs) == 0:
        return centroids
    for loc in locs:
        (startX, startY, endX, endY) = loc
        centroidX = (startX + endX)//2
        centroidY = (startY + endY)//2
        try:
            centroidZ = image[centroidY, centroidX]
        except:
            print(f"Cannot calculate centroidZ. loc: {loc}, image shape: {image.shape}")
            continue
        centroid = np.array([centroidX/w, centroidY/h, centroidZ/60000])
        centroids.append(centroid)
    return centroids

def euclide_distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)

def calculate_distance(args, all_locs, image_path_list):
    
    all_depths = predict_depths(args, natsorted(image_path_list))
    all_distances = []
    for locs, image_depth in tqdm(zip(all_locs, all_depths), total=len(image_path_list), desc="calculating distance"):
        centroids = calculate_centroid_coordination(image_depth, locs)
        min_distance = 100
        if len(centroids) in [0, 1]:
            all_distances.append(1)
            continue
        for i in range(len(centroids)):
            for j in range(i+1, len(centroids)):
                dist = euclide_distance(centroids[i], centroids[j])
                if dist < min_distance:
                    min_distance = dist
        all_distances.append(min_distance)
    return all_distances


if __name__ == '__main__':
    from parser_5k import set_parser
    import glob
    args = set_parser()
    image_path_list = glob.glob(f'{args.images_dir}/*.jpg')
    res = calculate_distance(args, image_path_list)
    print(res)