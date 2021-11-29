from tools.distance_calculate import calculate_distance
import pandas as pd
import os

df = pd.read_csv('data/train/train_meta.csv')
data_path = 'data/train/images'
mask_all = []
image_list = []
for idx, row in df.iterrows():
    sample_name = row["fname"]
    sample_path = os.path.join(data_path, sample_name)
    try:
        label_mask = int(row["mask"])
        image_list.append(sample_path)
        mask_all.append(label_mask)
    except:
        continue

print(f'length of mask_all: {len(mask_all)}')
print(f'length of image_list: {len(image_list)}')

load_ckpt_path = 'ckpt/res50.pth'
backbone = 'resnet50'
face_path = 'face_detect/models'
detect_confidence = 0.4

# calculate distance
all_distance = calculate_distance(load_ckpt_path, backbone, image_list, face_path, detect_confidence)

# determine gap
from tqdm import tqdm
min_yes = 100
max_no = 0
for dist, label in tqdm(zip(all_distance, mask_all), total=len(mask_all), desc="calculating gap"):
    if label == 1:
        if dist < min_yes:
            min_yes = dist
    else:
        if dist > max_no:
            max_no = dist

print("gap of min_yes and max no: {}".format(min_yes - max_no))
print("max_no value:", max_no)
print("min_yes value", min_yes)
print("means of 2 value", (min_yes + max_no)/2)    
