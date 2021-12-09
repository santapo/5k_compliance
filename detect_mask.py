import os
import glob

import cv2
import pandas as pd
import numpy as np

from face_detector.detector import RetinaFaceDetector
# from mask_classifier.deploy import classify

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

from tqdm import tqdm

if __name__ == "__main__":
    images_path = "data/images"
    image_path_list = glob.glob(os.path.join(images_path, "*"))
    
    # weigth_path = "face_detector/weights/Resnet50_Final.pth"
    detector = RetinaFaceDetector(is_visualize=False)

    with tf.device('/cpu:0'):
        mask_path = "mask_classifier/mask_detector.model"
        mask_classifier = load_model(mask_path)

    collected_data = []
    for i in tqdm(range(len(image_path_list))):
        image_path = image_path_list[i]
        image_name = os.path.split(image_path)[1]
        image = cv2.imread(image_path)
        # image = cv2.resize()
        image_height, image_width, _ = image.shape

        dets = detector.detect_face(image)
        
        for det in dets:
            start_x, start_y, end_x, end_y = det[0:4].astype("int")
            (start_x, start_y) = (max(0, start_x), max(0, start_y))
            face = image[start_y:end_y, start_x:end_x]
            # face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            # face = cv2.resize(face, (224, 224))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            with tf.device("/cpu:0"):
                (mask, withoutMask) = mask_classifier.predict(face)[0]
            
            data = (image_name, image_height, image_width, *list(map(int,det)), mask, withoutMask)
            collected_data.append(data)
        
        # if len(face_images) > 0:
        #     face_images = np.array(face_images, dtype="float32")
        #     preds = mask_classifier.predict(face_images, batch_size=8)
        #     import ipdb; ipdb.set_trace()
        #     # if visualize:
        #     #     visualize_image(image, locs, preds)
            # data = (image_name, image_height, image_width, )
            # collected_data.append((image_name, image_height, image_width, *list(map(int, det)), prob))

    
    dataframe = pd.DataFrame(collected_data, columns=['fname', 'image_height', 'image_width', 
                                'start_x', 'start_y', 'end_x', 'end_y', 'detect_conf', 'mask', 'withoutMask'])
    dataframe.to_csv('gnn_data.csv')

        #     prob_font_scale = (face.shape[0] * face.shape[1]) / (100 * 100)
        #     prob_font_scale = max(prob_font_scale, 0.25)
        #     prob_font_scale = min(prob_font_scale, 0.75)
        #     cv2.putText(image, '{0:.2f}'.format(prob), (start_x + 7, start_y - 3),
        #                 cv2.FONT_HERSHEY_SIMPLEX, prob_font_scale, (0, 0, 255), 1, lineType=cv2.LINE_AA)

        #     color = (0, 255, 0) if mask else (0, 211, 255)
        #     cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color, 2)
        # cv2.imshow("test", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
    

