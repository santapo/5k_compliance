from lib.multi_depth_model_woauxi import RelDepthModel
from lib.net_tools import load_ckpt
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import cv2
import os
import argparse
import numpy as np
import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Configs for LeReS')
    parser.add_argument('--load_ckpt', default='./res50.pth', help='Checkpoint path to load')
    parser.add_argument('--backbone', default='resnext101', help='Checkpoint path to load')

    args = parser.parse_args()
    return args

def scale_torch(img):
    """
    Scale the image and output it in torch.tensor.
    :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        transform = transforms.Compose([transforms.ToTensor(),
		                                transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225) )])
        img = transform(img)
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    return img

def predict_single_depth(depth_model, v):
    rgb = cv2.imread(v)
    rgb_c = rgb[:, :, ::-1].copy()
    gt_depth = None
    A_resize = cv2.resize(rgb_c, (448, 448))
    rgb_half = cv2.resize(rgb, (rgb.shape[1]//2, rgb.shape[0]//2), interpolation=cv2.INTER_LINEAR)

    img_torch = scale_torch(A_resize)[None, :, :, :]
    pred_depth = depth_model.inference(img_torch).cpu().numpy().squeeze()
    pred_depth_ori = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))

    # if GT depth is available, uncomment the following part to recover the metric depth
    #pred_depth_metric = recover_metric_depth(pred_depth_ori, gt_depth)
    depth = (pred_depth_ori/pred_depth_ori.max() * 60000).astype(np.uint16)
    return depth

def predict_depths(args, image_path_list):
    depth_model = RelDepthModel(backbone=args.backbone)
    depth_model.eval()
    print(image_path_list)
    # load checkpoint
    load_ckpt(args.load_ckpt, depth_model, None, None)
    depth_model.cuda()
    res_depth = []
    from tqdm import tqdm
    for img_path in tqdm(image_path_list, desc='predicting depth image'):
        image_depth = predict_single_depth(depth_model, img_path)
        res_depth.append(image_depth)

    return res_depth

if __name__ == "__main__":
    from parser_5k import set_parser
    import glob
    args = set_parser()
    image_path_list = glob.glob(f'{args.images_dir}/*.jpg')
    res = predict_depths(args, image_path_list)
    print(res)