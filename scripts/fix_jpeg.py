import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


VEG = 236
GRASS = 191
DIRT = 209
ROAD = 205
ROCKS = 255

classes = [GRASS, ROAD, DIRT, VEG, ROCKS]
high_borders = [
    GRASS + int(0.7 * (ROAD-GRASS)), # 200
    207,
    DIRT + int(0.3 * (VEG-DIRT)), # 217
    250
]

def clip(arr, values, borders):
    out = np.where(arr<=high_borders[0], values[0], arr)
    out = np.where(np.logical_and(arr>high_borders[0], arr<=high_borders[1]), values[1], out)
    out = np.where(np.logical_and(arr>high_borders[1], arr<=high_borders[2]), values[2], out)
    out = np.where(np.logical_and(arr>high_borders[2], arr<=high_borders[3]), values[3], out)
    out = np.where(arr>high_borders[3], values[4], out)
    assert len(np.unique(out)) <= 5
    return out

i = 0
data_dir = "/home/judicator/dozer/datasets/set_2"
for image_name in os.listdir(f"{data_dir}/masks"):
    i += 1
    if i % 100 == 0:
        print(i)
    image = cv2.imread(f"{data_dir}/masks/{image_name}")
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_np = np.array(image_gray)
    fixed = clip(image_np, classes, high_borders)
    new_name = f"{image_name.split('.')[0]}.png"
    cv2.imwrite(f"{data_dir}/masks_fixed/{new_name}", fixed)
