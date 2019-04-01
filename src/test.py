import numpy as np
import glob

from model_keras import nvidia_model
from augmentation import read_img
import rospkg
import cv2
def unwarp(img, src, dst):
        h, w = img.shape[:2]
        M = cv2.getPerspectiveTransform(src, dst)

        unwarped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
        return unwarped
def bird_view(source_img):
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
        unwarped = unwarp(source_img, src, dst)
        return unwarped
def test():
    path = rospkg.RosPack().get_path('team105')
    print(path)
    model = nvidia_model()
    model.load_weights(path + '/param/cds-weights.09-0.02990.h5')
    model._make_predict_function()

    for img_path in glob.glob(path + '/test/*.jpg'):
        test_img = read_img(img_path)

        middle_pos = img_path.split('_')[4]

        test_img = np.array(test_img)
        predicted_middle_pos = float(model.predict(bird_view(test_img)[None, :, :, :], batch_size=1)) * 80 + 160
        print(middle_pos, predicted_middle_pos)
if __name__ == '__main__':
    test()
