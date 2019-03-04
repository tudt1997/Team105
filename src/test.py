import numpy as np
import glob

from model_keras import nvidia_model
from augmentation import read_img
import rospkg
import rospy

if __name__ == '__main__':
    rospy.init_node('team105', anonymous=True)
    path = rospkg.RosPack().get_path('team105')

    model = nvidia_model()
    model.load_weights(path + '/param/new-weights.03-0.36097.h5')
    model._make_predict_function()


    for img_path in glob.glob(path + '/test/*.jpg'):
        test_img = read_img(img_path)

        middle_pos = img_path.split('_')[4]

        test_img = np.array(test_img)
        predicted_middle_pos = float(model.predict(test_img[None, :, :, :], batch_size=1)) * 80 + 160
        print(middle_pos, predicted_middle_pos)