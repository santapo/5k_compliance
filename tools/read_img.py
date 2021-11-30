import os, glob, shutil
import cv2
import numpy as np

def visualize_image(image_path, flag: str = 'color'):
    if flag == 'grey':
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(image_path)
    cv2.imshow(f"{os.path.basename(image_path)}", img)
    cv2.waitKey(0)

def draw_rectangle(image, locs):
    for loc in locs:
        startX, startY, endX, endY = loc
        image = cv2.rectangle(image, (startX, startY), (endX, endY), (0, 128, 0), 3)
    cv2.imshow("bounding box", image)
    cv2.waitKey(0)