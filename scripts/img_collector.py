import datetime
import pathlib
import argparse
import numpy as np
import cv2
import roslib
import rospy
import message_filters
from sensor_msgs.msg import CompressedImage



class ImageCollector:

    def __init__(self, topic_name, save_folder, rate, verbose):
        '''Initialize ros publisher, ros subscriber'''
        self.topic_name = topic_name
        self.verbose = verbose
        self.save_path = pathlib.Path(save_folder)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.rate = int(rate)
        self.img_counter = 0
        # self.subscriber = rospy.Subscriber(
        #     f"{topic_name}", CompressedImage, self.callback,  queue_size=1
        # )
        # if self.verbose :
        #     print(f"subscribed to {topic_name}")

    def callback(self, ros_data):
        '''Callback function of subscribed topic'''
        if self.img_counter % self.rate == 0:
            if self.verbose :
                print('received image of type: "%s"' % ros_data.format)
            np_arr = np.fromstring(ros_data.data, np.uint8)
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            filename = f"{self.img_counter + 1}.{ros_data.format}"
            cv2.imwrite(str(self.save_path / filename), image_np)
        self.img_counter += 1


def main(args):
    '''Initializes and cleanup ros node'''
    ic1 = ImageCollector(args.image_topic, args.save_folder + "/train", args.rate, args.verbose)
    ic2 = ImageCollector(args.mask_topic, args.save_folder + "/train_mask", args.rate, args.verbose)
    image_sub = message_filters.Subscriber(args.image_topic, CompressedImage)
    mask_sub = message_filters.Subscriber(args.mask_topic, CompressedImage)
    ts = message_filters.TimeSynchronizer([image_sub, mask_sub], 1)
    
    def ts_callback(image, mask):
        ic1.callback(image)
        ic2.callback(mask)

    ts.registerCallback(ts_callback)

    rospy.init_node(f'ImageCollector', anonymous=True)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS ImageCollector module")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_topic', default='/sim/camera/compressed')
    parser.add_argument('-m', '--mask_topic', default='/sim/segmentation/compressed')
    parser.add_argument('-f', '--save_folder', default=folder)
    parser.add_argument('-v', '--verbose', default=True)
    parser.add_argument('-r', '--rate', default=10)
    args = parser.parse_args()
    main(args)