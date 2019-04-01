#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32, String, Bool
import cv2
import math
import rospkg
import cv2
import numpy as np
from model_keras import nvidia_model

path = rospkg.RosPack().get_path('team105_detectsign')
from augmentation import read_img
import glob
import keras

import time
import tensorflow as tf

print(tf.__version__)
print(keras.__version__)
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))


class car_control:
    def __init__(self):
        rospy.init_node('team105', anonymous=True)
        self.speed_pub = rospy.Publisher("/set_speed_car_api", Float32, queue_size=1)
        self.steerAngle_pub = rospy.Publisher("/set_steer_car_api", Float32, queue_size=1)
        self.lcd_pub = rospy.Publisher("/lcd_print", String, queue_size=4)

        rospy.Rate(10)

        self.stopping = True
        self.option = 0
        self.base_speed = 10
        self.current_speed = 0
        self.speed_decay = 0.1
        self.steer_angle_scale = 1.25
        self.middle_pos_scale = 1
        self.choose_symbol = ['>', ' ', ' ', ' ']

        self.line1 = '0:0:{1}speed={0:2}'
        self.line2 = '0:1:{1}decay={0:4}'
        self.line3 = '0:2:{1}steer_sc={0:4}'
        self.line4 = '0:3:{1}mid_pos_sc={0:4}'
        self.publish_lcd()

        # Load keras model
        self.model = nvidia_model()
        self.model.load_weights(path + '/param/mar30_final_2-old-weights.04-0.02966.h5')
        self.model._make_predict_function()
        self.h, self.w = 240, 320
        self.carPos = (160, 240)
        self.last_detected = 0
        self.sign_type = 0
        self.is_turning = False
        self.time_detected = 0
        # self.test(self.model)
        self.fps = 0
        self.fps_timer = time.time()

        self.cover_left = np.array([[0, 0], [200, 0], [100, 240]])
        self.cover_right = np.array([[120, 0], [320, 0], [220, 240]])
        self.triangle_cnt = np.array([[0,0]])
        self.b1 = False
        self.b2 = False
        self.b3 = False
        self.b4 = False

        self.sub_b1 = rospy.Subscriber("/bt1_status", Bool, callback=self.callback1)
        self.sub_b2 = rospy.Subscriber("/bt2_status", Bool, callback=self.callback2)
        self.sub_b3 = rospy.Subscriber("/bt3_status", Bool, callback=self.callback3)
        self.sub_b4 = rospy.Subscriber("/bt4_status", Bool, callback=self.callback4)

    def callback1(self, data):
        self.b1 = data.data

    def callback2(self, data):
        self.b2 = data.data

    def callback3(self, data):
        self.b3 = data.data

    def callback4(self, data):
        self.b4 = data.data

    def check_button(self):
        if self.b1:
            if self.option == 0:
                self.base_speed = min(25, self.base_speed + 1)
            if self.option == 1:
                self.speed_decay = min(0.5, self.speed_decay + 0.05)
            if self.option == 2:
                self.steer_angle_scale = min(2, self.steer_angle_scale + 0.05)
            if self.option == 3:
                self.middle_pos_scale = min(2, self.middle_pos_scale + 0.05)
            print('bt1')
            self.publish_lcd()

        elif self.b2:
            if self.option == 0:
                self.base_speed = max(10, self.base_speed - 1)
            if self.option == 1:
                self.speed_decay = max(0.05, self.speed_decay - 0.05)
            if self.option == 2:
                self.steer_angle_scale = max(1, self.steer_angle_scale - 0.05)
            if self.option == 3:
                self.middle_pos_scale = max(1, self.middle_pos_scale - 0.05)
                
            print('bt2')
            self.publish_lcd()

        elif self.b3:
            self.option = (self.option + 1) % 4
            self.publish_lcd()
            print('bt3')

        elif self.b4:
            self.option = (self.option - 1 + 4) % 4
            self.publish_lcd()
            print('bt4')

    def control(self, img, sign, isGo, danger_zone):
        if not rospy.is_shutdown():


            steerAngle = self.cal_steerAngle(img, sign, danger_zone) * self.steer_angle_scale
            #print(steerAngle, isGo)
            if self.current_speed >= 10 and self.current_speed < self.base_speed:
                self.current_speed += 0.5

            speed = np.max(np.array([10, self.current_speed - self.speed_decay * abs(steerAngle)])) 
            # High steer angle

            if isGo == False:
                self.stopping = True
                self.current_speed = 0
                steerAngle = 0
                speed = 0
            elif self.stopping:
                self.stopping = False
                self.current_speed = 10
            print(self.current_speed)
            #print(steerAngle, speed)
            self.speed_pub.publish(speed)
            self.steerAngle_pub.publish(-steerAngle)
        return self.is_turning, steerAngle, speed

    def cal_steerAngle(self, img, sign, danger_zone):
        # fps counter
        self.fps += 1
        if time.time() - self.fps_timer > 1:
            print("fps ", self.fps)
            self.fps_timer = time.time()
            self.fps = 0
        steerAngle = 0

        if sign != 0:
            self.time_detected = time.time()
            # turn right
            if sign == 1:
                self.triangle_cnt = self.cover_left
            # turn left
            else:
                self.triangle_cnt = self.cover_right

        img_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_bv = self.bird_view(img_array)

        # cover in 1 sec
        if time.time() - self.time_detected < 1:
            cv2.drawContours(img_bv, [self.triangle_cnt], 0, (0, 0, 0), -1)
        # timer = time.time()
        middle_pos = float(self.model.predict(img_bv[None, :, :, :], batch_size=1)) * 160 + 160

        
        if danger_zone != (0, 0):
            center_danger_zone = int((danger_zone[0] + danger_zone[1]) / 2)
            #print(danger_zone, center_danger_zone)
            if middle_pos > danger_zone[0]+50 and middle_pos < danger_zone[1]-50:
                # obstacle's on the right
                if middle_pos < center_danger_zone:
                    print("on the right")
                    middle_pos = danger_zone[0]
                # left
                else:
                    print("on the left")
                    middle_pos = danger_zone[1]
        # print("drive ",time.time()-timer)

        if middle_pos > 640:
            middle_pos = 640
        if middle_pos < -320:
            middle_pos = -320

        cv2.line(img_bv, (int(middle_pos), self.h / 2), (self.w / 2, self.h), (255, 0, 0), 2)
        cv2.imshow("Bird view", img_bv[:,:,:])
        # Distance between MiddlePos and CarPos
        distance_x = middle_pos - self.w / 2
        distance_y = self.h - self.h / 2

        # Angle to middle position
        steerAngle = math.atan(float(distance_x) / distance_y) * 180 / math.pi
        cv2.waitKey(1)

        return steerAngle

    def publish_lcd(self):
        self.lcd_pub.publish(self.line1.format(self.base_speed, self.choose_symbol[self.option]))
        self.lcd_pub.publish(self.line2.format(self.speed_decay, self.choose_symbol[self.option - 1]))
        self.lcd_pub.publish(self.line3.format(self.steer_angle_scale, self.choose_symbol[self.option - 2]))
        self.lcd_pub.publish(self.line4.format(self.middle_pos_scale, self.choose_symbol[self.option - 3]))

    def unwarp(self, img, src, dst):
        h, w = img.shape[:2]
        M = cv2.getPerspectiveTransform(src, dst)

        unwarped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
        return unwarped

    def bird_view(self, source_img):
        h, w = source_img.shape[:2]
        # define source and destination points for transform

        src = np.float32([(100, 120),
                          (220, 120),
                          (0, 210),
                          (320, 210)])

        dst = np.float32([(120, 0),
                          (w - 120, 0),
                          (120, h),
                          (w - 120, h)])

        # change perspective to bird's view
        unwarped = self.unwarp(source_img, src, dst)
        return unwarped
