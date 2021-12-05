
import os
import glob
import cv2
import numpy as np
import logging

import torch
import torch.backends.cudnn as cudnn

from face_detector.config import cfg_re50
from face_detector.layers.functions.prior_box import PriorBox
from face_detector.utils.nms.py_cpu_nms import py_cpu_nms
from face_detector.utils.box_utils import decode, decode_landm
from face_detector.models.retinaface import RetinaFace


from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model as tf_load_model
from utils import visualize_image
import numpy as np
import argparse
import logging
import cv2
import glob
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detect_mask_images")


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def detect_face(image, face_detector, cfg, detect_confidence):
    device = torch.device("cpu")
    top_k = 5000
    keep_top_k = 750
    nms_threshold = 0.4
    resize = 1

    image = np.float32(image)

    # testing scale
    target_size = 1600
    max_size = 2150
    im_shape = image.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    resize = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(resize * im_size_max) > max_size:
        resize = float(max_size) / float(im_size_max)
    image = cv2.resize(image, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

    im_height, im_width, _ = image.shape
    scale = torch.Tensor([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
    image -= (104, 117, 123)
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).unsqueeze(0)
    image = image.to(device)
    scale = scale.to(device)

    loc, conf, landms = face_detector(image)  # forward pass
    

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([image.shape[3], image.shape[2], image.shape[3], image.shape[2],
                            image.shape[3], image.shape[2], image.shape[3], image.shape[2],
                            image.shape[3], image.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > detect_confidence)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)
    
    return dets
    # show image
    # if args.save_image:
    #     for b in dets:
    #         if b[4] < args.vis_thres:
    #             continue
    #         text = "{:.4f}".format(b[4])
    #         b = list(map(int, b))
    #         cv2.rectangle(image_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
    #         cx = b[0]
    #         cy = b[1] + 12
    #         cv2.putText(img_raw, text, (cx, cy),
    #                     cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

    #         # landms
    #         cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
    #         cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
    #         cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
    #         cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
    #         cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
    #     # save image

    #     name = "test.jpg"
    #     cv2.imwrite(name, img_raw)

    # faces = []
    # locs = []
    # preds = []
    # for i in range(0, detections.shape[2]):
    #     confidence = detections[0, 0, i, 2]
    #     if confidence > detect_confidence:
    #         box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    #         (startX, startY, endX, endY) = box.astype("int")

    #         # Ensure the bounding boxes fall within the dimensions of the frame
    #         (startX, startY) = (max(0, startX), max(0, startY))
    #         (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

    #         # Extract the face ROI, convert it from BGR to RGB channel
    #         # ordering, resize it to 224x224, and preprocess it
    #         face = image[startY:endY, startX:endX]
    #         if face.any():
    #             face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    #             face = cv2.resize(face, (224, 224))
    #             face = img_to_array(face)
    #             face = preprocess_input(face)
                
    #             faces.append(face)
    #             locs.append((startX, startY, endX, endY))
    
    # only make a prediction if at least one face was detected
    # if len(faces) > 0:
    #     faces = np.array(faces, dtype="float32")
    #     preds = mask_classifier.predict(faces, batch_size=32)

    # return (locs, preds)

def detect_mask_images(image_path_list, face_path, mask_path, detect_confidence, visualize=False):
    cfg = cfg_re50
    face_detector = RetinaFace(cfg=cfg, phase='test')
    face_detector = load_model(face_detector, face_path, load_to_cpu=True)
    face_detector.eval()
    logger.info("Finished Loading RetinaFace!")

    mask_classifier = tf_load_model(mask_path)
    logger.info("Finished Loading mask classification model!")

    res_list = []
    for image_path in image_path_list:
        print(image_path)
        image_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        faces = detect_face(image_raw, face_detector, cfg, detect_confidence)

        face_images = []
        face_locs = []
        for i in range(faces.shape[0]): 
            (startX, startY, endX, endY) = faces[i, 0:4].astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            face = image_raw[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            # face = img_to_array(face)
            # face = preprocess_input(face)
            face_images.append(face)

            face_locs.append((startX, startY, endX, endY))
        # if image_path == "data/public_test/images/466.jpg":
        #     import ipdb; ipdb.set_trace()
        #     cv2.imwrite("test.jpg", faces[0])
        # import ipdb; ipdb.set_trace()
        if len(face_images) > 0:
            face_images = np.array(face_images, dtype="float32")
            preds = mask_classifier.predict(face_images, batch_size=4)
            # if visualize:
            #     visualize_image(image, locs, preds)
            res_list.append((face_locs, preds, image_path))
            # import ipdb; ipdb.set_trace()
    return res_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images_dir", type=str, default=None,
                        help="Path to images directory")
    parser.add_argument("-f", "--face_weight", type=str, default="face_detector/model",
                        help="Path to face detection model directory")
    parser.add_argument("-m", "--mask_weight", type=str, default="mask_classifier/mask_detector.model",
                        help="Path to mask classification model directory")
    parser.add_argument("-dc", "--detect_confidence", type=float, default=0.5,
                        help="Minimum probability to filer weak detections")
    args = vars(parser.parse_args())

    logger.info("Loading input images...")
    image_path_list = glob.glob(os.path.join(args["images_dir"], "*"))
    res_list = detect_mask_images(image_path_list, args["face_weight"], args["mask_weight"], args["detect_confidence"], visualize=True)

    




