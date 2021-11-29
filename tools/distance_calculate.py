import cv2
import numpy as np
import torch
from face_detect.face import detect_faces
from tools.test_depth import predict_depths
from tqdm import tqdm

def calculate_centroid_coordination(image, locs):
    (h, w) = image.shape[:2]
    centroids = []
    if len(locs) == 0:
        return centroids
    for loc in locs:
        (startX, startY, endX, endY) = loc
        centroidX = (startX + endX)//2
        centroidY = (startY + endY)//2
        centroidZ = image[centroidY, centroidX]
        centroid = np.array([centroidX, centroidY, centroidZ]) / np.array([w, h, 60000])
        centroids.append(centroid)
    return centroids

def euclide_distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)

def calculate_distance(load_ckpt, backbone,image_path_list, face_path, detect_confidence):
    all_locs = detect_faces(image_path_list, face_path, detect_confidence)
    all_depths = predict_depths(load_ckpt, backbone, image_path_list)
    all_distances = []
    for idx, (locs, image_depth) in tqdm(enumerate(zip(all_locs, all_depths)), total=len(image_path_list), desc="calculating distance"):
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
    load_ckpt_path = 'ckpt/res50.pth'
    backbone = 'resnet50'
    import glob
    image_path_list = glob.glob('./data/*')
    face_path = 'face_detect/models'
    detect_confidence = 0.4
    all_distances = calculate_distance(load_ckpt_path, backbone, image_path_list, face_path, detect_confidence)
    
    print(all_distances)