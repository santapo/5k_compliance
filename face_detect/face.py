import cv2 
import numpy as np  
import logging
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detect face in image")

def detect_single_object(image, face_detector, detect_confidence):
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(640, 640),
                                mean=(104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()

    locs = []
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
            
            locs.append((startX, startY, endX, endY))
    
    # only make a prediction if at least one face was detected

    return locs

def detect_faces(image_path_list, face_path, detect_confidence, visualize=False):
    logger.info("Loading face detection model...")
    prototxt_path = os.path.join(face_path, "deploy.prototxt")
    weights_path = os.path.join(face_path, "res10_300x300_ssd_iter_140000.caffemodel")
    detector = cv2.dnn.readNet(prototxt_path, weights_path)

    res_list = []
    from tqdm import tqdm
    for image_path in tqdm(image_path_list, desc='detecting faces'):
        image = cv2.imread(image_path)
        locs = detect_single_object(image, detector, detect_confidence)
        res_list.append(locs)

    return res_list

def visualize_pred_box(image_path, detector, detect_confidence, output_path = None):
    
    image = cv2.imread(image_path)
    locs = detect_single_object(image, detector, detect_confidence)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for loc in locs:
        (startX, startY, endX, endY) = loc
        image = cv2.rectangle(image, (startX, startY), (endX, endY), (0, 128, 0), 1)
    if output_path:
        cv2.imwrite(image, output_path)
    else:
        print(locs)
        cv2.imshow(f"{os.path.basename(image_path)}", image)
        cv2.waitKey(0)


if __name__ == "__main__":
    import glob
    image_path_list = glob.glob('data/*')
    face_path = 'face_detect/models'
    detect_confidence = 0.4
    prototxt_path = os.path.join(face_path, "deploy.prototxt")
    weights_path = os.path.join(face_path, "res10_300x300_ssd_iter_140000.caffemodel")
    detector = cv2.dnn.readNet(prototxt_path, weights_path)
    image = cv2.imread('data/3.jpg')
    res = detect_single_object(image, detector, detect_confidence)
    print(image.shape)
    image = cv2.rectangle(image, (res[3][0], res[3][1]), (res[3][2], res[3][3]), (0, 128, 0), 1)
    cv2.imshow(" ", image)
    cv2.waitKey(0)
