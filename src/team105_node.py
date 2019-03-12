#!/usr/bin/env python
# from __future__ import print_function
import roslib

roslib.load_manifest('team105')
import sys
import rospy
import cv2
from std_msgs.msg import Float32, Bool
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import matplotlib.pyplot as plt
from car_control import car_control

import time


class image_converter:
    def __init__(self):
        self.isGo = True
        self.go_sub = rospy.Subscriber("/ss_status", Bool, callback=self.isAGo, queue_size=1)


        self.cc = car_control()
        self.ld = lane_detector()
        self.curr_time = "100"
        rospy.Rate(10)
        # self.sd = SignDetection()
        self.is_turning = False

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("camera/rgb/image_raw/compressed", CompressedImage, callback=self.callback, queue_size=1)

    def isAGo(self,data):
        self.isGo = data.data

    def callback(self, data):
        try:
            np_arr = np.fromstring(data.data, np.uint8)
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # NOTE: image_np.shape = (240,320,3)
            image_np = cv2.resize(image_np, (320, 240))
            sign = 0
            # drive

            self.is_turning, steer_angle, speed = self.cc.control(image_np, sign, self.isGo)
            
        except CvBridgeError as e:
            print(e)

if __name__ == '__main__':
    rospy.init_node('team105', anonymous=True)
    ic = image_converter()
    rospy.spin()
