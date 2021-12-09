
import os
import glob
import cv2
import numpy as np
import logging

import torch
import torch.backends.cudnn as cudnn

from .config import cfg_re50
from .layers.functions.prior_box import PriorBox
from .utils.nms.py_cpu_nms import py_cpu_nms
from .utils.box_utils import decode
from .models.retinaface import RetinaFace

class RetinaFaceDetector():
    def __init__(self,
                 weight_path: str = "/home/santapo/OnlineLab/challenges/5k_compliance_zalo/5k_compliance/face_detector/weights/Resnet50_Final.pth",
                 device: str = "cpu",
                 config: dict = cfg_re50,
                 is_visualize: bool = False,
                 target_size: int = 1600,
                 max_size: int = 2150,
                 top_k: int = 5000,
                 keep_top_k: int = 750,
                 nms_threshold: float = 0.2,
                 detect_confidence: float = 0.5
                 ):

        self.device = torch.device(device)
        load_to_cpu = True if device == "cpu" else False
        
        self.config = config
        self.is_visualize = is_visualize
        self.target_size = target_size
        self.max_size = max_size
        self.top_k = top_k
        self.keep_top_k = keep_top_k
        self.nms_threshold = nms_threshold
        self.detect_confidence = detect_confidence

        self.model = self._load_model(RetinaFace(cfg=self.config, phase="test"),
                                        weight_path, load_to_cpu).eval()
        
    
    def _check_keys(self, model, pretrained_state_dict):
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

    def _remove_prefix(self, state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def _load_model(self, model, pretrained_path, load_to_cpu):
        print('Loading pretrained model from {}'.format(pretrained_path))
        if load_to_cpu:
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self._remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self._remove_prefix(pretrained_dict, 'module.')
        self._check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model
    
    def visualize(self, image, dets):
        for b in dets:
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(image, text, (cx, cy),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
            cv2.imshow('test', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def resize_image(self, image):
        image_size_min = np.min(image.shape[0:2])
        image_size_max = np.max(image.shape[0:2])
        resize = float(self.target_size) / float(image_size_min)
        if np.round(resize * image_size_max) > self.max_size:
            resize = float(self.max_size) / float(image_size_max)
        if resize != 1:
            image = cv2.resize(image, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        return resize, image

    def preprocess_image(self, image):
        scale = torch.Tensor([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
        image -= (104, 117, 123)
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).unsqueeze(0)
        image = image.to(self.device)
        scale = scale.to(self.device)
        return scale, image

    def detect_face(self, raw_image):
        image = np.float32(raw_image)
        resize, image = self.resize_image(image)
        image_height, image_width, _ = image.shape
        scale, image = self.preprocess_image(image)
        # import ipdb; ipdb.set_trace()

        loc, conf, _ = self.model(image)  # forward pass
        

        priorbox = PriorBox(self.config, image_size=(image_height, image_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.config['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > self.detect_confidence)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        
        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]

        if self.is_visualize:
            self.visualize(raw_image, dets)
        return dets


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-i", "--images_dir", type=str, default=None,
    #                     help="Path to images directory")
    # parser.add_argument("-f", "--face_weight", type=str, default="face_detector/model",
    #                     help="Path to face detection model directory")
    # parser.add_argument("-m", "--mask_weight", type=str, default="mask_classifier/mask_detector.model",
    #                     help="Path to mask classification model directory")
    # parser.add_argument("-dc", "--detect_confidence", type=float, default=0.5,
    #                     help="Minimum probability to filer weak detections")
    # args = vars(parser.parse_args())

    # logger.info("Loading input images...")
    images_path = "/home/santapo/OnlineLab/challenges/5k_compliance_zalo/5k_compliance/data/public_test/images"
    image_path_list = glob.glob(os.path.join(images_path, "*"))
    
    weigth_path = "/home/santapo/OnlineLab/challenges/5k_compliance_zalo/5k_compliance/face_detector/weights/Resnet50_Final.pth"
    detector = RetinaFaceDetector(is_visualize=True)

    image = cv2.imread(image_path_list[0])
    detector.detect_face(image)

    




