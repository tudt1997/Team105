#!/usr/bin/env python
# from __future__ import print_function
import roslib

roslib.load_manifest('team105')
import sys
import rospy
import cv2
from skimage import img_as_float
from std_msgs.msg import Float32, Bool
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import matplotlib.pyplot as plt
from car_control import car_control
from sign_detection import sign_detection
from compressedDepth import depth_camera

import time


class image_converter:
    def __init__(self):
        self.isGo = True
        self.go_sub = rospy.Subscriber("/ss_status", Bool, callback=self.isAGo, queue_size=1)

        self.cc = car_control()
        self.sd = sign_detection()
        self.dc = depth_camera()

        self.sd_counter = 0
        self.cc_counter = 0
        self.dc_counter = 0
        self.bt_counter = 0

        self.sign = 0
        self.bbox_obstacles = []
        self.danger_zone = (0, 0)

        self.curr_time = "100"
        rospy.Rate(10)

        self.is_turning = False

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("camera/rgb/image_raw/compressed", CompressedImage, callback=self.callback,
                                          queue_size=1)
        self.image_detect_sign_sub = rospy.Subscriber("camera/rgb/image_raw/compressed", CompressedImage, callback=self.callback_detectsign, queue_size=1)

        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, callback=self.callback_detectobstacle, queue_size=1)
       
        self.fps_counter_test = 0
        self.fps_timer = time.time()

    def isAGo(self, data):
        self.isGo = data.data

    def callback(self, data):
        try:
            self.cc_counter+=1
            if self.cc_counter % 2 == 0:
                self.cc_counter=0

                np_arr = np.fromstring(data.data, np.uint8)
                image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                #image_np = self.bridge.imgmsg_to_cv2(data)

                # NOTE: image_np.shape = (240,320,3)
                image_np = cv2.resize(image_np, (320, 240))
                # drive

                self.is_turning, steer_angle, speed = self.cc.control(image_np, self.sign, self.isGo, self.danger_zone)      
      
            self.bt_counter += 1
            if self.bt_counter % 10 == 0:
                self.bt_counter = 0
                self.cc.check_button()
        except CvBridgeError as e:
            print(e)

    def callback_detectsign(self, data):
        try:
            self.sd_counter += 1
            if self.sd_counter % 5 == 0:
                self.sd_counter = 0
                np_arr = np.fromstring(data.data, np.uint8)
                image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                #image_np = self.bridge.imgmsg_to_cv2(data)

                img_out, sign_list, p_list = self.sd.detect_sign(image_np)
                # cv2.imshow("sign_detection", img_out)
                # cv2.waitKey(1)

                # always get the max P of list
                if len(sign_list) > 0:
                    index_max = np.argmax(np.array(p_list)[:, 1])
                    self.sign = sign_list[index_max][1]

                else:
                    self.sign = 0

        except CvBridgeError as e:
            print(e)


    def callback_detectobstacle(self, data):
        
        try:
            self.dc_counter += 1
            if self.dc_counter % 3 == 0:
                self.dc_counter = 0

                self.fps_counter_test += 1
                if time.time() - self.fps_timer > 1:
                    print(self.fps_counter_test)
                    self.fps_counter_test = 0
                    self.fps_timer = time.time()

                #image_np = self.dc.process_compressedDepth(data)
                image_np = self.bridge.imgmsg_to_cv2(data)

                image_np = cv2.resize(image_np, (320, 240))
                image_np = image_np[100:,:]

                #cv2.imshow('img_depth', image_np)
                #cv2.waitKey(1)
                # timer = time.time()
                self.bbox_obstacles = self.dc.pre_processing_depth_img(image_np*10)
                # timer = time.time() - timer
                # print("time: ", timer)
                #print(time.time() - timer)
                if len(self.bbox_obstacles) > 0:
                    index = np.argmax(np.array(self.bbox_obstacles)[:, 1])
                    nearest_obstacle = self.bbox_obstacles[index]
                    (x, y, w, h) = nearest_obstacle
                    self.danger_zone = (x - 100, x + w + 100)
                else:
                    self.danger_zone = (0, 0)
                
                #print(self.danger_zone)

        except CvBridgeError as e:
            print(e)


if __name__ == '__main__':
    rospy.init_node('team105', anonymous=True)
    ic = image_converter()
    rospy.spin()
