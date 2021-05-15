import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt

data_dir = '/home/judicator/dozer/datasets/set_2'
i = 0

VEG = 236
GRASS = 191
DIRT = 209
ROAD = 205
ROCKS = 255

pix_tresh = 2500

for mask_name in os.listdir(f'{data_dir}/masks_fixed'):
    i+=1
    if i%100 == 0:
        print(i)
    image = cv2.imread(f'{data_dir}/masks_fixed/{mask_name}', cv2.IMREAD_GRAYSCALE)
    image_np = np.array(image)

    if ((image_np == DIRT).sum() == 0 and
        ((image_np == ROAD).sum() > pix_tresh or
         (image_np == ROCKS).sum() > pix_tresh)):
        a = mask_name.split('.')[0]
        img_name = f'{a}.jpeg'
        shutil.copy(f'{data_dir}/images/{img_name}', f'{data_dir}/images_no_dirt')
        shutil.copy(f'{data_dir}/masks_fixed/{mask_name}', f'{data_dir}/masks_no_dirt')

print(len(os.listdir(f'{data_dir}/images_no_dirt')))
