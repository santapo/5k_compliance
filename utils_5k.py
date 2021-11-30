import cv2
import pandas as pd
from natsort import natsorted
import glob
import json

def visualize_image(image, locs, preds):
    for box, pred in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, without_mask) = pred

        label = "Mask" if mask > without_mask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        label = "{}: {:.2f}%".format(label, max(mask, without_mask) * 100)

        cv2.putText(image, label, (startX, startY -10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        
    image = cv2.resize(image, (400, 300))
    cv2.imshow("Prediction", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def list_of_txt_to_locs(locs_dir):
    txt_list = natsorted(glob.glob(f'{locs_dir}/*.txt'))
    all_locs = []
    for file in txt_list:
        single_image_locs = []
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                single_image_locs.append(json.loads(line))
        all_locs.append(single_image_locs)
    return all_locs

def merge_result_to_meta(meta_path, result_list):
    meta = pd.read_csv(meta_path)
    result_df = pd.DataFrame(result_list, columns=["fname", "5K"])
    merged_df = pd.merge(meta, result_df, on="fname")
    return merged_df

