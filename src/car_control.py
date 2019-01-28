import rospy
from std_msgs.msg import Float32
import cv2
import math
import time
from time import sleep

class car_control:
    def __init__(self, team_name):
        rospy.init_node('team105', anonymous=True)
        self.speed_pub = rospy.Publisher(team_name + "_speed", Float32, queue_size=1)
        self.steerAngle_pub = rospy.Publisher(team_name + "_steerAngle", Float32, queue_size=1)
        rospy.Rate(10)

        self.carPos = (160, 240)
        self.last_detected = 0
        self.sign_type = 0
        self.is_turning = False

    def control(self, sign, middlePos):
        if not rospy.is_shutdown():
            now = time.time()
            diff = now - self.last_detected

            steerAngle = self.cal_steerAngle(sign, middlePos, diff)
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


    def cal_steerAngle(self, sign, middlePos, diff):
        carPos_x, carPos_y = self.carPos

        middlePos_x, middlePos_y = middlePos

        steerAngle = 0

        if (sign[4] > 20):
            self.last_detected = time.time()
            self.sign_type = 1
<<<<<<< HEAD
        if (sign[4] < -25):
=======
        if (sign[4] < -20):
>>>>>>> f30d20dd586631234018ed2dfbb4e8e255c7c10f
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
            # Distance between MiddlePos and CarPos
            distance_x = middlePos_x - carPos_x
            distance_y = carPos_y - middlePos_y

            # Angle to middle position
            steerAngle = math.atan(distance_x / distance_y) * 180 / math.pi
            # print(middlePos_x, steerAngle)

        return steerAngle

