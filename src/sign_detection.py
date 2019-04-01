from tinyyolo import TinyYoloNet
from labeling import label_anno
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import numpy as np
import time
import os
from nonmax_supression import nonmax_supression
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import sign_classi
import rospkg

class sign_detection:
    def __init__(self):
        path = rospkg.RosPack().get_path('team105_detectsign')

        self.sign = ['nope', 'turn right', 'turn left']

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Tiny YOLO
        self.net = TinyYoloNet()
        self.net.load_state_dict(torch.load(path + '/param/param7_epoch1.txt'))
        print('loaded model YOLO')
        self.net.eval()
        self.net.to(self.device).float()
        self.net.cuda()

        self.generic_transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
        ])

        self.sc = sign_classi.Net()

    def detect_sign(self,img_np):
        input = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        # to PIL
        input = Image.fromarray(input)

        # to tensor
        input = self.generic_transform(input).to(self.device).float().cuda()

        # add 1 more dimension
        input = input.view(1, 3, 448, 448)

        # detect sign
        time_start = time.time()
        output = self.net(input)
        time_finish = time.time()
        #print("time : {}".format(time_finish - time_start))

        output = output.view(5, 13, 13)
        output = output.cpu().detach().numpy()
        output = np.transpose(output, (1,2,0))

        list_bbox_p = label_anno(img_np, output)
        supressed_bboxs = nonmax_supression(list_bbox_p)

        # container
        rec_list = []
        p_list = []
        sign_list = []
        for bb in supressed_bboxs:
            # to int
            x1, y1, x2, y2 , p= int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]), bb[4]

            crop = img_np[y1:y2, x1:x2, :]
            crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

            # push crop image into classify net / classify sign (nope-right-left)
            classify_output = self.sc.predict(crop)
            if classify_output != 0:
                #TO DO: Slow down if sign detected
				if x2-x1 > 55 and y2-y1 > 55:
                	#cv2.imshow('crop', crop)
                	#cv2.waitKey(1)
            		rec_list.append(((x1,y1),(x2,y2)))
            		p_list.append(((x1,y1), p))
            		sign_list.append(((x1,y1-13), classify_output))

        img_out = np.copy(img_np)
        # draw to image
        for i in range(len(rec_list)):
            cv2.rectangle(img_out, rec_list[i][0], rec_list[i][1], (0, 255, 0), 3)
            cv2.putText(img_out, "%.3f" % p_list[i][1], p_list[i][0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 2, cv2.CV_AA)
            cv2.putText(img_out, self.sign[sign_list[i][1]], sign_list[i][0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,0,0), 2, cv2.CV_AA)

        return img_out, sign_list, p_list

