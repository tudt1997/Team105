import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

class lane_detector:
    def __init__(self):
        self.laneWidth = 0

    # return the smaller
    def cal_quadratic_smaller(self, a, b, c):
        delta = b**2. - 4.*a*c
        if delta < 0 or a == 0:
            # print("delta",delta)
            return float('inf')
        else:
            y1 = (-b - math.sqrt(delta))/(2.*a)
            y2 = (-b + math.sqrt(delta))/(2.*a)

            if 0 <= y1 <= 240:
                return y1
            return y2

    # transform to bird-view
    def unwarp(self, img, src, dst):
        h, w = img.shape[:2]
        M = cv2.getPerspectiveTransform(src, dst)

        unwarped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
        return unwarped


    # transform from bird-view to normal view
    def warp(self, src_img, line_img, src, dst):
        h,w = line_img.shape[:2]
        M = cv2.getPerspectiveTransform(src, dst)

        warped = cv2.warpPerspective(line_img, M, (w,h))
        result = cv2.addWeighted(src_img, 1, warped, 0.5, 0)
        return result


    # detect white line (210, 240)
    def white_thresh(self, img, threshold_white=(210, 240), threshold_white_shadow=(140, 143)):
        # using lightness channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        # detect white line (210, 240)
        hls_l = hls[:, :, 1]
        hls_l = hls_l * (255 / np.max(hls_l))
        bin_white = np.zeros_like(hls_l)
        bin_white[(hls_l > threshold_white[0]) & (hls_l <= threshold_white[1])] = 255

        # cv2.imshow("white", bin_white)
        # cv2.waitKey(1)
        # detect white lane in shadow (140,143)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab_b = lab[:, :, 2]
        bin_white_shadow = np.zeros_like(lab_b)
        bin_white_shadow[(lab_b >= threshold_white_shadow[0]) & (lab_b <= threshold_white_shadow[1])] = 255

        # cv2.imshow("shadow", bin_white_shadow)
        # cv2.waitKey(1)

        bin_out = np.zeros_like(lab_b)
        bin_out[(bin_white == 255) | (bin_white_shadow == 255)] = 255

        return bin_out


    def restrict_similar_function(self, left_fit, right_fit, left_lane_inds, right_lane_inds):
        # restrict wrong detection
        left_check = self.cal_quadratic_smaller(left_fit[0], left_fit[1], left_fit[2] - 159.5)
        right_check = self.cal_quadratic_smaller(right_fit[0], right_fit[1], right_fit[2] - 159.5)

        # print("left_y", left_check)
        # print("right_y", right_check)

        checksum = float('inf')
        if (left_check != float('inf') and right_check != float('inf')):
            checksum = math.fabs(left_check - right_check)

        # print("checksum_full" , checksum)
        # 2 lanes have similar function
        if (checksum <= 100):
            # print("checksum", checksum)

            min_nonzero_left_y = left_lane_inds[0]
            min_nonzero_right_y = right_lane_inds[0]

            if min_nonzero_left_y > min_nonzero_right_y:
                right_fit = [0, 0, 0]
                right_lane_inds = []
                # print("remove right")

            else:
                left_fit = [0, 0, 0]
                left_lane_inds = []
                # print("remove left")

        return left_fit, right_fit, left_lane_inds, right_lane_inds, checksum


    def sliding_window_polyfit(self, img):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(img, axis=0)
        midpoint = np.int(histogram.shape[0] // 2)
        # Previously the left/right base was the max of the left/right half of the histogram
        leftx_base = np.argmax(histogram[20:midpoint-20]) + 20 # margin sliding window
        rightx_base = np.argmax(histogram[midpoint+20:2*midpoint]) + midpoint + 20

        # print('base pts:', leftx_base, rightx_base)

        # Choose the number of sliding windows
        nwindows = 10
        # Set height of windows
        window_height = np.int(img.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 20
        # Set minimum number of pixels found to recenter window
        minpix = 20
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        # Rectangle data for visualization
        rectangle_data = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            rectangle_data.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high))
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                        nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                        nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # must have >= 150 pixels to fit polynomial
        if len(left_lane_inds) < 150:
            left_lane_inds = []
        if len(right_lane_inds) < 150:
            right_lane_inds = []

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_fit = [0, 0, 0]
        right_fit = [0, 0, 0]
        # Fit a second order polynomial to each

        if len(leftx) != 0:
            left_fit = np.polyfit(lefty, leftx, 2)
        if len(rightx) != 0:
            right_fit = np.polyfit(righty, rightx, 2)

        visualization_data = (rectangle_data, histogram)


        return left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data


    def find_middlePos(self, left_fit, right_fit, checksum, is_turning):
        # middlePos_y = 120
        # middlePos_y2 = 240

        middlePos_y = np.array([120, 240])

        # print(self.laneWidth)
        # Detect nothing
        if left_fit[0] == 0 and right_fit[0] == 0:
            return (-1, -1, -1, -1)
        # Detect right side
        elif left_fit[0] == 0 and right_fit[0] != 0:
            rightSide_x = right_fit[0] * middlePos_y ** 2 + right_fit[1] * middlePos_y + right_fit[2]

            middlePos_x = rightSide_x - self.laneWidth/2
        # Detect left side
        elif left_fit[0] != 0 and right_fit[0] == 0:
            leftSide_x = left_fit[0] * middlePos_y ** 2 + left_fit[1] * middlePos_y + left_fit[2]

            middlePos_x = leftSide_x + self.laneWidth/2
        # Detect both
        else:
            leftSide_x = left_fit[0] * middlePos_y ** 2 + left_fit[1] * middlePos_y + left_fit[2]
            rightSide_x = right_fit[0] * middlePos_y ** 2 + right_fit[1] * middlePos_y + right_fit[2]

            if(checksum > 100 and not is_turning):
                self.laneWidth = rightSide_x[0] - leftSide_x[0]

            # print("WIDTH ", rightSide_x - leftSide_x)
            middlePos_x = (leftSide_x + rightSide_x) / 2

        result = np.concatenate((middlePos_x, middlePos_y))
        return result


    def lane_detect(self, source_img, is_turning):
        cv2.setUseOptimized(True)
        h, w = source_img.shape[:2]
        # define source and destination points for transform

        src = np.float32([(100, 120),
                          (220, 120),
                          (0, 210),
                          (320, 210)])

        dst = np.float32([(80, 0),
                          (w-80, 0),
                          (80, h),
                          (w-80, h)])

        # change perspective to bird's view
        unwarped = self.unwarp(source_img, src, dst)

        # cv2.imshow("unwarp", unwarped)
        # cv2.waitKey(1)

        # detect white + white_shadow
        bin_out = self.white_thresh(unwarped)

        # dilate
        kernel = np.ones((3, 3), np.uint8)
        bin_out = cv2.dilate(bin_out, kernel,iterations=1)

        # cv2.imshow("white + shadow", bin_out)
        # cv2.waitKey(1)

        # Detect lines
        lines = cv2.HoughLinesP(bin_out, 2, np.pi / 180, 128, minLineLength=78,maxLineGap=10)

        houghline_img = np.zeros_like(bin_out)
        if lines is not None:
            for i in range(0, len(lines)):
                l = lines[i][0]
                cv2.line(houghline_img, (l[0], l[1]), (l[2], l[3]), 255, 2, cv2.LINE_AA)

        # cv2.imshow("Detect_line", houghline_img)
        # cv2.waitKey(1)

        # AND bin_houghline & bin_out
        bin_line_only = np.zeros_like(bin_out)
        bin_line_only[(houghline_img == 255) & (bin_out == 255)] = 1

        # cv2.imshow("Lines only", bin_line_only*255)
        # cv2.waitKey(1)

        left_fit, right_fit, left_lane_inds, right_lane_inds, \
        visualization_data = self.sliding_window_polyfit(bin_line_only)

        left_fit, right_fit, \
        left_lane_inds, right_lane_inds, checksum = self.restrict_similar_function(left_fit, right_fit, left_lane_inds, right_lane_inds)

        # print(left_fit, right_fit)
        rectangles = visualization_data[0]

        middlePos = self.find_middlePos(left_fit, right_fit, checksum, is_turning)
        # print(middlePos)

        # draw middle line
        line_img = np.zeros_like(source_img)
        pts1 = (int(middlePos[0]), int(middlePos[2]))
        pts2 = (int(middlePos[1]), int(middlePos[3]))
        cv2.line(line_img, pts1, pts2, (0, 255, 0), 2, cv2.LINE_AA)
        out_img = self.warp(source_img, line_img, dst, src)


#         # Create an output image to draw on and  visualize the result
#         lane_img = np.uint8(np.dstack((bin_line_only, bin_line_only, bin_line_only)) * 255)

#         try:
#             # Generate x and y values for plotting
#             ploty = np.linspace(0, bin_line_only.shape[0] - 1, bin_line_only.shape[0]//2)
#             left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
#             right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

#             pts1 = np.vstack((left_fitx, ploty)).astype(np.int32).T
#             pts2 = np.vstack((right_fitx, ploty)).astype(np.int32).T
#             # Plot
#             cv2.polylines(lane_img, [pts1], False, (0,255,255), 1)
#             cv2.polylines(lane_img, [pts2], False, (0,255,255), 1)
#         except:
#             pass
#         # Draw the windows on the visualization image
#         for rect in rectangles:
#             cv2.rectangle(lane_img, (rect[2], rect[0]), (rect[3], rect[1]), (0, 255, 0), 2)
#             cv2.rectangle(lane_img, (rect[4], rect[0]), (rect[5], rect[1]), (0, 255, 0), 2)

#         # Identify the x and y positions of all nonzero pixels in the image
#         nonzero = bin_line_only.nonzero()
#         nonzeroy = np.array(nonzero[0])
#         nonzerox = np.array(nonzero[1])
#         # Change color of nonzero pixels
#         lane_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
#         lane_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # cv2.imshow("Detect lane", lane_img)
        # cv2.waitKey(1)

        return out_img, middlePos

