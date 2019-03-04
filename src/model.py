import torch.nn as nn

class NvidiaNet(nn.Module):
    def __init__(self):
        super(NvidiaNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2),
            nn.MaxPool2d(2, 2),
            nn.ELU(inplace=True),
            # nn.BatchNorm2d(24),

            nn.Conv2d(24, 36, kernel_size=5, padding=2),
            nn.MaxPool2d(2, 2),
            nn.ELU(inplace=True),
            # nn.BatchNorm2d(36),

            nn.Conv2d(36, 48, kernel_size=5, padding=2),
            nn.MaxPool2d(2, 2),
            nn.ELU(inplace=True),
            # nn.BatchNorm2d(48),

            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            # nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            # nn.BatchNorm2d(64)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4608, 200),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(200),

            nn.Linear(200, 50),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(50),

            nn.Linear(50, 1)
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
