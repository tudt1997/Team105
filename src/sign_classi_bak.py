import torch as tr, cv2, numpy as np
import rospkg
device = tr.device("cuda" if tr.cuda.is_available() else "cpu")
path = rospkg.RosPack().get_path('team105_detectsign')
class NET(tr.nn.Module):
    def __init__(self):
        super(NET, self).__init__()  # 3x32x32

        self.conv = tr.nn.Sequential(
            tr.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            tr.nn.ReLU(),
            tr.nn.MaxPool2d(2, 2),

            tr.nn.Conv2d(32, 64, 3, 2, 1),
            tr.nn.ReLU(),
            tr.nn.MaxPool2d(2, 2),

            # tr.nn.Conv2d(64, 128, 3, 1, 1),
            # tr.nn.ReLU(),
            # tr.nn.MaxPool2d(2, 2),
        )

        self.fc1 = tr.nn.Sequential(
            tr.nn.Linear(64*4*4, 30),
            tr.nn.ReLU(),
        )

        self.fc2 = tr.nn.Sequential(
            tr.nn.Linear(30, 3),
            tr.nn.Softmax(dim=1),
        )

    def forward(self, x):
        # drop = tr.nn.Dropout(droprate)

        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # x = drop(x)
        x = self.fc2(x)
        return x

net= NET().to(device)
net.load_state_dict(tr.load(path + '/param/param_newdata1'))

def predict(img):
    new_size = 32
    img = cv2.resize(img, (new_size, new_size))

    if img.max()>1:
        img = np.array(img, dtype= np.float32) / 255.

    img= img.reshape(1,new_size,new_size,3).transpose((0,3,1,2))

    with tr.no_grad():
        img = tr.from_numpy(img).to(device)
        output= net(img)
        output= tr.argmax(output)

    return output.item() #0= not turn; 1= turn right, 2= turn left

if __name__=="__main__":
    import os
    with tr.no_grad():
        for file in os.listdir('other imgs'):
            img= cv2.imread('other imgs/'+file)
            # img= cv2.imread('adddata_neg/neg (102).jpg')
            print(predict(img))
            cv2.imshow(str(predict(img)), cv2.resize(img, (150,150)))
            cv2.waitKey(0)
