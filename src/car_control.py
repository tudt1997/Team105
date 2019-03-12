#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32
import cv2
import math
import rospkg
import cv2
import numpy as np
from model_keras import nvidia_model
path = rospkg.RosPack().get_path('team105')
from augmentation import read_img
import glob
import keras
from sign_detection import detect_sign

import tensorflow as tf
print(tf.__version__)
print(keras.__version__)
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

class car_control:
    def __init__(self):
        rospy.init_node('team105', anonymous=True)
        self.speed_pub = rospy.Publisher("/set_speed_car_api", Float32, queue_size=1)
        self.steerAngle_pub = rospy.Publisher("/set_steer_car_api", Float32, queue_size=1)
        rospy.Rate(10)

        # Load keras model
        self.model = nvidia_model()
        self.model.load_weights(path + '/param/cds-weights.09-0.02990.h5')
        self.model._make_predict_function()
        self.h, self.w = 240, 320
        self.carPos = (160, 240)
        self.last_detected = 0
        self.sign_type = 0
        self.is_turning = False
        # self.test(self.model)

    def control(self, img, sign, isGo):
        if not rospy.is_shutdown():
            steerAngle = self.cal_steerAngle(img, sign) * 1.5
            print(steerAngle, isGo)

            speed = 12
            # High steer angle
            if math.fabs(steerAngle) >= 30:
                speed = 12

            if isGo == False:
                steerAngle = 0
                speed = 0

            # print(middlePos[0])
            self.speed_pub.publish(speed)
            self.steerAngle_pub.publish(-steerAngle)

        return self.is_turning, steerAngle, speed


    def cal_steerAngle(self, img, sign):
        steerAngle = 0

        img_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	# detect sign
	img_out, _ = detect_sign(img_array)

        img_bv = self.bird_view(img_array)
        middle_pos = float(self.model.predict(img_bv[None, :, :, :], batch_size=1)) * 160 + 160

        if middle_pos > 320:
            middle_pos = 320
        if middle_pos < 0:
            middle_pos = 0

        cv2.line(img_bv, (int(middle_pos), self.h / 2), (self.w / 2, self.h), (255, 0, 0), 2)
        cv2.imshow("Bird view", img_bv)
        # Distance between MiddlePos and CarPos
        distance_x = middle_pos - self.w / 2
        distance_y = self.h - self.h / 2

        # Angle to middle position
        steerAngle = math.atan(float(distance_x) / distance_y) * 180 / math.pi

        return steerAngle

    def test(self, model):

        for img_path in glob.glob(path + '/test/*.jpg'):
            test_img = read_img(img_path)

            middle_pos = float(img_path.split('_')[-2]) * 160 + 160

            test_img = np.array(test_img)
            predicted_middle_pos = float(model.predict(test_img[None, :, :, :], batch_size=1)) * 160 + 160

            print(middle_pos, predicted_middle_pos)

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
