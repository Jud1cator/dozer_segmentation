import datetime
import pathlib
import argparse
import numpy as np
import cv2
import roslib
import rospy
import cv_bridge

from sensor_msgs.msg import CompressedImage, Image
from message_filters import ApproximateTimeSynchronizer, Subscriber


class DatasetCollector:
    def __init__(self):
        self.counter = 0
        self.rate = 10
        folder = f"/home/judicator/dozer/datasets/set_2"
        self.img_path = pathlib.Path(folder + "/images")
        self.mask_path = pathlib.Path(folder + "/masks")
        self.img_path.mkdir(parents=True, exist_ok=True)
        self.mask_path.mkdir(parents=True, exist_ok=True)

    def save_image(self, image, name, path):
        bridge = cv_bridge.CvBridge()
        img = bridge.imgmsg_to_cv2(image, "bgr8")
        # np_arr = np.frombuffer(image.data, np.uint8)
        # image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        filename = f"{name}.png"
        cv2.imwrite(str(path / filename), img)

    def save_compressed_image(self, image, name, path):
        np_arr = np.fromstring(image.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        filename = f"{name}.{image.format}"
        cv2.imwrite(str(path / filename), image_np)

    def callback(self, image, mask):
        if self.counter % self.rate == 0:
            name = self.counter // self.rate + 1
            self.save_compressed_image(image, name, self.img_path)
            self.save_image(mask, name, self.mask_path)
        self.counter += 1


rospy.init_node("node", anonymous=True)
dc = DatasetCollector()
tss = ApproximateTimeSynchronizer(
    # [Subscriber("/sim/camera/compressed", CompressedImage),
    #  Subscriber("/sim/segmentation/compressed", CompressedImage)],
    # [Subscriber("/sim/camera/image_raw", Image),
    #  Subscriber("/sim/segmentation/image_raw", Image)],
    [Subscriber("/sim/camera/compressed", CompressedImage),
     Subscriber("/sim/segmentation/image_raw", Image)],
    1,
    0.1
)
tss.registerCallback(dc.callback)
print('Subscribed to topics')
rospy.spin()
