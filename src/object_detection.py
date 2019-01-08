import cv2
import numpy as np
from yolo import predict

def detect_object(image_np):
    img = image_np[:, :]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    threshold = cv2.inRange(hsv,(48,0,0),(255,255,73))
    combined = threshold
    # combined = cv2.GaussianBlur(combined, (5, 5), 0)
    combined = cv2.blur(combined, (3, 3))
    # rev = cv2.bitwise_not(combined)
    # cv2.imshow("Thresholding", rev)


    # image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    sign_x = sign_y = sign_w = sign_h = sign_size = 0
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(threshold, kernel, iterations=1)
    cntr_frame, contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('image', erosion)
    cv2.waitKey(1)
    for cnt in contours:
        [x, y, w, h] = cv2.boundingRect(cnt)

        if (w > 10 and h > 10 and 0.8 < h / w < 1.0 / 0.8):

            pred = predict(img)

            if pred != 3:
                # print(pred)
                sign_x = x
                sign_y = y
                sign_w = w
                sign_h = h
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                if pred == 0:
                    cv2.putText(img, 'Pile of barrel', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    sign_size = w
                elif pred== 1:
                    cv2.putText(img, 'Boulder', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    sign_size = -w
                elif pred== 2:
                    cv2.putText(img, 'Barrel', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    sign_size = -w
                break

    return img
