import rospy
from std_msgs.msg import Float32
import cv2
import math
import time
from time import sleep
# import torchvision.transforms as transforms
# from model import *
from PIL import Image
# import torch
import rospkg
import cv2
import numpy as np
from keras.models import load_model
from model_keras import nvidia_model
path = rospkg.RosPack().get_path('team105')
from augmentation import read_img
import glob
# import test
import keras
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import tensorflow as tf
print(tf.__version__)
print(keras.__version__)
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

class car_control:
    def __init__(self, team_name):
        rospy.init_node('team105', anonymous=True)
        self.speed_pub = rospy.Publisher(team_name + "_speed", Float32, queue_size=1)
        self.steerAngle_pub = rospy.Publisher(team_name + "_steerAngle", Float32, queue_size=1)
        rospy.Rate(10)

        # Load pytorch model
        # self.net = NvidiaNet().to(device)
        # self.net.load_state_dict(torch.load(path + '/param/model_yuv_1550497211.4968872_1.4583244496025145', map_location=device))

        # self.transform = transforms.Compose([
        #     transforms.Resize((100, 200)),
        #     transforms.ToTensor()]
        # )
        self.i = 0
        # Load keras model
        self.model = nvidia_model()
        self.model.load_weights(path + '/param/new-weights.03-0.36097.h5')
        self.model._make_predict_function()
        self.test(self.model)
        # checkpoint = torch.load(path + '/param/model3.h5', map_location=lambda storage, loc: storage)
        # self.net = checkpoint['net'].to(device)
        self.carPos = (160, 240)
        self.last_detected = 0
        self.sign_type = 0
        self.is_turning = False

    def control(self, sign, middlePos, img):
        if not rospy.is_shutdown():
            now = time.time()
            diff = now - self.last_detected

            steerAngle = self.cal_steerAngle(sign, middlePos, diff, img)
            print(steerAngle)

            speed = 50
            # cant detect 2 lanes
            if (middlePos[0] == -1 or math.fabs(steerAngle) >= 10) and diff > 2:
                speed = 30
            elif math.fabs(steerAngle) >= 5 or (diff > 0.1 and diff < 2):
                speed = 40

            # print(middlePos[0])
            self.speed_pub.publish(speed)
            self.steerAngle_pub.publish(steerAngle)

        return self.is_turning, steerAngle, speed


    def cal_steerAngle(self, sign, middlePos, diff, img):
        carPos_x, carPos_y = self.carPos

        middlePos_x, middlePos_y = middlePos

        steerAngle = 0

        if (sign[4] > 20):
            self.last_detected = time.time()
            self.sign_type = 1
        if (sign[4] < -20):
            self.last_detected = time.time()
            self.sign_type = -1

        if (diff > 0.35 and diff < 0.5):
            self.is_turning = True

        if self.is_turning:
            if diff < 1.5:
                steerAngle = 50 * self.sign_type
            else:
                if middlePos_x == -1:
                    self.last_detected = time.time() - 1.5
                    steerAngle = 50 * self.sign_type
                else:
                    self.is_turning = False
                    steerAngle = 0
        elif middlePos_x == -1:
            steerAngle = 0
        else:
            # print(middlePos_x, steerAngle)

            # img = np.array(img, dtype='>f') / 255.
            # img = cv2.resize(img[80:, :, :], (150, 200))
            # img = img.reshape(1, 150, 200, 3).transpose((0, 3, 1, 2))
                # img = torch.from_numpy(img).to(device)
                # img = cv2.cvtColor(, (150, 200))
            img_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_array = np.array(img_array)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            # img = Image.fromarray(img[80:, :, :])
            # img = self.transform(img).to(device)
            # img = img.view(1, 3, 100, 200)
            # image = torch.autograd.Variable(img)

            # middle_pos = self.net(img).view(-1).data.cpu().numpy()[0] * 320
            # middle_pos = float(self.model.predict(img_array[None, :, :, :], batch_size=1)) * 80 + 160
            middle_pos = float(self.model.predict(np.expand_dims(img_array, axis=0), batch_size=1)) * 80 + 160
            if middle_pos > 320:
                middle_pos = 320
            if middle_pos < 0:
                middle_pos = 0
            # Distance between MiddlePos and CarPos
            distance_x = middle_pos - carPos_x
            distance_y = carPos_y - middlePos_y

            print(middlePos_x, middle_pos, middlePos_x - middle_pos)
            # Angle to middle position
            steerAngle = math.atan(distance_x / distance_y) * 180 / math.pi

        return steerAngle

    def test(self, model):

        for img_path in glob.glob(path + '/test/*.jpg'):
            test_img = read_img(img_path)

            middle_pos = img_path.split('_')[4]

            test_img = np.array(test_img)
            predicted_middle_pos = float(model.predict(test_img[None, :, :, :], batch_size=1)) * 80 + 160

            # predicted_middle_pos = float(self.model.predict(np.expand_dims(test_img, axis=0), batch_size=1)) * 80 + 160

            print(middle_pos, predicted_middle_pos)
