import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model import SemSegment
from torchvision import transforms as transforms


class_weights = np.array([0.13130568, 1., 0.75924934, 0.7089291])
class_weights = torch.Tensor(class_weights).cuda()
model = SemSegment(
    num_classes=4,
    num_layers=5,
    features_start=16,
    class_weights=class_weights
)
model.load_state_dict(torch.load('./models/model_set2.json'))
model.eval()

# data_dir = '/home/judicator/dozer/datasets/2021-04-06_19:03:46'
data_dir = '/home/judicator/dozer/datasets/set_2'
img_folder = 'images'
mask_folder = 'masks_fixed'


img = "28.jpeg"
test_img = Image.open(f'{data_dir}/{img_folder}/{img}')
test_mask = Image.open(f'{data_dir}/masks/{img.split(".")[0]}.png')
test_img = test_img.resize((1920, 1080))
test_img = np.array(test_img)
# plt.imshow(test_img)
# plt.show()
test_input = transforms.ToTensor()(test_img).unsqueeze(0)

with torch.no_grad():
    test_out = model(test_input)
    test_out = np.array(test_out)

out_img = np.zeros((1080, 1920, 3), dtype='uint8')

classes = (191, 205, 236, 255)
classes_rgb = [
    [162,212,163],
    [220,193,224],
    [218,255,183],
    [255,255,255]
]
class_map = dict(zip(range(len(classes)), classes))

for i in range(1080):
    for j in range(1920):
        out_img[i, j] = classes_rgb[np.argmax(test_out[0, :, i, j], 0)]

# save = Image.fromarray(out_img)
# save.save(f'{data_dir}/predictions/{img}')

plt.imshow(test_img)
plt.show()
plt.imshow(test_mask, cmap='gray')
plt.show()
plt.imshow(out_img)
plt.show()
