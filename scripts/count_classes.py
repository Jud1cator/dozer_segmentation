import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

GRASS = 191
ROAD = 205
DIRT = 209
VEG = 236
ROCKS = 255

classes = [GRASS, ROAD, DIRT, VEG, ROCKS]
total_classes = [0 for i in range(len(classes))]
n = 0

data_dir = "/home/judicator/dozer/datasets/set_2"
for image_name in os.listdir(f"{data_dir}/masks_no_dirt"):
    n += 1
    if n % 100 == 0:
        print(n)
    image = cv2.imread(f"{data_dir}/masks_no_dirt/{image_name}", cv2.IMREAD_GRAYSCALE)
    image_np = np.array(image)
    # unique = np.unique(image_np)
    # print({u: (image_np == u).sum() for u in unique})
    # input()
    for i in range(len(classes)):
        total_classes[i] += (image_np == classes[i]).sum()

total_classes = np.array(total_classes)
print(total_classes)
weights = total_classes.sum() / total_classes
weights = weights / weights.max()
print(weights)
