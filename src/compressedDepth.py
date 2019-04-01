#!/usr/bin/env python
import cv2
import numpy as np
import time
import rospy
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge, CvBridgeError
import math
import struct
import rospkg



class depth_camera:
    def __init__(self):
        self.SCALE = 10000  # ==1m
        self.MAX_UINT16 = 65536
        self.counter = 0
        #self.path = rospkg.RosPack.get_path('team105_detectsign')
        

    def process_compressedDepth(self, msg):
        # 'msg' as type CompressedImage
        depth_fmt, compr_type = msg.format.split(';')
        # remove white space
        depth_fmt = depth_fmt.strip()
        compr_type = compr_type.strip()
        if compr_type != "compressedDepth":
            raise Exception("Compression type is not 'compressedDepth'."
                            "You probably subscribed to the wrong topic.")

        # remove header from raw data
        depth_header_size = 12
        raw_data = msg.data[depth_header_size:]

        depth_img_raw = cv2.imdecode(np.fromstring(raw_data, np.uint8), cv2.IMREAD_UNCHANGED)
        if depth_img_raw is None:
            # probably wrong header size
            raise Exception("Could not decode compressed depth image."
                            "You may need to change 'depth_header_size'!")

        if depth_fmt == "16UC1":
            # write raw image data
            print('16UC1')
            return depth_img_raw
            # cv2.imwrite(os.path.join(path_depth, "depth_" + str(msg.header.stamp) + ".png"), depth_img_raw)
        elif depth_fmt == "32FC1":
            raw_header = msg.data[:depth_header_size]
            # header: int, float, float
            [compfmt, depthQuantA, depthQuantB] = struct.unpack('iff', raw_header)
            depth_img_scaled = depthQuantA / (depth_img_raw.astype(np.float32) - depthQuantB)
            # filter max values
            depth_img_scaled[depth_img_raw == 0] = 0

            # depth_img_scaled provides distance in meters as f32
            # for storing it as png, we need to convert it to 16UC1 again (depth in mm)
            depth_img_mm = (depth_img_scaled * self.SCALE).astype(np.uint16)
            # cv2.imwrite(os.path.join(path_depth, "depth_" + str(msg.header.stamp) + ".png"), depth_img_mm)
            return depth_img_mm

        else:
            raise Exception("Decoding of '" + depth_fmt + "' is not implemented!")

    def ground(self, gray_img, x, y, n, T1, T2):
        if gray_img[y][x] > T1 or gray_img[y][x] < T2:
            return 0
        if int(gray_img[y][x]) - int(gray_img[y+n][x]) >= 1:
            for i in range(n):
                if int(gray_img[y+i][x]) - int(gray_img[y+i+1][x]) < 0:
                    return gray_img[y][x]
            return 0
        else:
            return gray_img[y][x]


    def resize_np(self, img_np, percent):
        h, w = img_np.shape
        w = int(w * percent)
        h = int(h * percent)
        resized_img = cv2.resize(img_np, (w, h))
        return resized_img


    def pre_processing_depth_img(self, img_np, n=2, T1=13000, T2=1000, min_width=30, min_height=30):
        #cv2.imwrite("./img_depth/raw/src_depth_"+str(self.counter)+".jpg", img_np)
        # resize
        gray_img = self.resize_np(img_np, 0.125)
        # cv2.imshow('src', gray_img)

        # CLOSE
        kernel_close = np.ones((3, 3))
        gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel_close)

        # DILATE
        kernel_dilate = np.ones((3, 3))
        gray_img = cv2.dilate(gray_img, kernel_dilate)
        #print(np.max(gray_img))
        #cv2.imshow('depth', gray_img)
        # cv2.waitKey()

        height, width = gray_img.shape
        # print(height, width)

        # remove floor and wall far away...
        for x in range(width):
            for y in range(height):
                 #if gray_img[y][x] > T1 or gray_img[y][x] < T2:
                 #    gray_img[y][x] = 0
                if y < height - n:
                    gray_img[y][x] = self.ground(gray_img, x, y, n, T1, T2)
                else:
                    gray_img[y][x] = 0

        #cv2.imshow('after remove floor', gray_img)
        
        # OPEN
        kernel_open = np.ones((3, 3), np.uint8)
        gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel_open)
        #cv2.imshow('removed_ground_OPEN', gray_img)
        
        # CLOSE
        #kernel_close = np.ones((3, 3), np.uint8)
        #gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel_close)
        #cv2.imshow('removed_ground_CLOSE', gray_img)

        # resize
        gray_img = self.resize_np(gray_img, 8)
        #cv2.imwrite("/img_depth/img_processed/processed_"+str(self.counter)+".jpg", gray_img)
        cv2.imshow('preprocessed', gray_img)
        
        #ret, thresh = cv2.threshold(gray_img, 10, 200, cv2.THRESH_BINARY)
        #cv2.imshow('bin', thresh)
        #print(np.max(gray_img))
        gray_uint8 = cv2.convertScaleAbs(gray_img)
        contours, hierarchy = cv2.findContours(gray_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        img_RGB_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        #img_RGB_np = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
        # cv2.drawContours(img_RGB_np, contours, -1, (MAX_UINT16, 0, 0), 2)
        # cv2.imshow('contours', img_RGB_np)
        bbox = []
        #print('number of contours', len(contours))
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            
            # print(w,h)
            if w > min_width and h > min_height:
                cv2.rectangle(img_RGB_np, (x, y), (x + w, y + h), (0, self.MAX_UINT16, 0), 2)
                # draw danger zone
                cv2.rectangle(img_RGB_np, (x-100, y), (x + w + 100, y + h), (self.MAX_UINT16, 0, 0), 2)
                bbox.append((x, y, w, h))
        cv2.imshow('box', img_RGB_np)
        cv2.waitKey(1)
        return bbox


if __name__ == '__main__':
    rospy.init_node('team105', anonymous=True)
    dc = depth_camera()
    rospy.spin()
