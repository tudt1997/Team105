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
path = rospkg.RosPack().get_path('team105')

sign = ['nope', 'turn right', 'turn left']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Tiny YOLO
net = TinyYoloNet()
net.load_state_dict(torch.load(path + '/param/yolo_param4_epoch4.txt'))
print('loaded model YOLO')
net.eval()
net.to(device).float()
net.cuda()

# CLASSIFY MODEL
# classify_net = CNN32()
# classify_net.load_state_dict(torch.load('./param_classify/sign_classi_param32_small'))
# classify_net.eval()
# classify_net.to(device).float()
# classify_net.cuda()

generic_transform = transforms.Compose([
        transforms.Resize((448,448)),
        transforms.ToTensor(),
])


def detect_sign(img_np):
    input = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    # to PIL
    input = Image.fromarray(input)

    # to tensor
    input = generic_transform(input).to(device).float().cuda()

    # add 1 more dimension
    input = input.view(1, 3, 448, 448)

    # detect sign
    time_start = time.time()
    output = net(input)
    time_finish = time.time()
    print("time : {}".format(time_finish - time_start))

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

        # push crop image into classify net / classify sign (nope-right-left)
        classify_output = sign_classi.predict(crop)
        # if classify_output != 0:
        #     cv2.imshow('crop', crop)
            # cv2.waitKey()

        rec_list.append(((x1,y1),(x2,y2)))
        p_list.append(((x1,y1), p))
        sign_list.append(((x1,y1-13), sign[classify_output]))

    img_out = np.copy(img_np)
    # draw to image
    for i in range(len(rec_list)):
        cv2.rectangle(img_out, rec_list[i][0], rec_list[i][1], (0, 255, 0), 3)
        cv2.putText(img_out, "%.3f" % p_list[i][1], p_list[i][0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 2, cv2.CV_AA)
        cv2.putText(img_out, sign_list[i][1], sign_list[i][0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,0,0), 2, cv2.CV_AA)

    return img_out, rec_list

def img_test():
    ########## TEST ON IMAGE ##########
    imgPath = './test_image/'

    imgIter = iter(os.listdir(imgPath))

    for _ in range(len(os.listdir(imgPath))):

        # for _ in range(13):
        imgFile = imgIter.__next__()


        img_np = cv2.imread(imgPath + imgFile)
        img_out, _ = detect_sign(img_np)
        cv2.imshow('img', img_out)
        cv2.waitKey()


def video_test():
    ####### TEST ON VIDEO #########
    import cv2

    cap = cv2.VideoCapture('D:/DATASET/MOV_0222.mp4')

    if cap.isOpened() is False:
        print("Error opening video stream or file")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = detect_sign(frame)
            # Display the resulting frame
            cv2.imshow('Frame', frame)
            # cv2.waitKey()

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img_test()
