#!/usr/bin/env python
# from __future__ import print_function
import roslib

roslib.load_manifest('team105')
import sys
import rospy
import cv2
from std_msgs.msg import Float32
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import matplotlib.pyplot as plt
from car_control import car_control
from sign_detection import SignDetection
from lane_detector import lane_detector
from object_detection import detect_object
import time
TEAM_NAME = 'team105'


class image_converter:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(TEAM_NAME + "_image/compressed", CompressedImage,
                                          callback=self.callback, queue_size=1)
        self.cc = car_control(TEAM_NAME)
        self.ld = lane_detector()
        self.curr_time = "100"
        rospy.Rate(10)
        self.sd = SignDetection()
        self.i = 0
        self.is_turning = False

    def callback(self, data):
        # print("callback")
        try:
            np_arr = np.fromstring(data.data, np.uint8)
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # NOTE: image_np.shape = (240,320,3)

            out_img, sign = self.sd.detect_sign(image_np)
            # out_img = image_np
            # sign = (0, 0, 0, 0, 0)
            # sign_size = 0
            #cv2.imshow("Object", img_object)
            #cv2.waitKey(1)

            # image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            out_img, middlePos = self.ld.lane_detect(out_img, self.is_turning)
            # print(middlePos)
            # print("Left ",left_fit," Right ",right_fit)

            cv2.imshow("Middle Pos", out_img)
            #img_object = detect_object(out_img)
            #cv2.imshow("Detect Object", img_object)
            cv2.waitKey(1)

            # drive
            if self.i == 0:
                self.is_turning,steer_angle,speed = self.cc.control(sign, (middlePos[0], middlePos[2]), image_np)
                self.i = 0
            else:
                self.i += 1
            # milliseconds = int(round(time.time() * 1000))
            #
            # if int(self.curr_time) <= milliseconds:
            #     file_name = "/home/nda97531/catkin_ws/src/team105/CDS/" + self.curr_time + "_"+ str(steer_angle)+"_"+ str(speed)+ ".jpg"
            #     self.curr_time = str(milliseconds+300)
            #     cv2.imwrite(file_name, image_np)

        except CvBridgeError as e:
            print(e)

if __name__ == '__main__':
    rospy.init_node('team105', anonymous=True)
    # cv2.namedWindow("houghlines")
    # def nothing():
    #     pass
    # cv2.createTrackbar("rho", "houghlines",2,255,nothing)
    # cv2.createTrackbar("theta", "houghlines",180,255,nothing)
    # cv2.createTrackbar("minLine", "houghlines",78,255,nothing)
    # cv2.createTrackbar("maxGap", "houghlines",10,255,nothing)
    # cv2.waitKey(1)
    ic = image_converter()
    rospy.spin()
