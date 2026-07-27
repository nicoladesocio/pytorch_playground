import torch

class InceptionNaive(torch.nn.Module):
    def __init__(self, in_channels, out_channels_array):
        super(InceptionNaive, self).__init__()
        self.branch1x1 = torch.nn.Conv2d(in_channels, out_channels_array[0], kernel_size=1)
        self.branch3x3 = torch.nn.Conv2d(in_channels, out_channels_array[1], kernel_size=3, padding=1)
        self.branch5x5 = torch.nn.Conv2d(in_channels, out_channels_array[2], kernel_size=5, padding=2)
        self.branch_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        return torch.cat([branch1x1, branch3x3, branch5x5, branch_pool], 1)

class Inception(torch.nn.Module):
    def __init__(self, in_channels, out_channels_array):
        super(Inception, self).__init__()
        self.branch1x1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels_array[0], kernel_size=1),
            torch.nn.ReLU(inplace=True)
        )
        self.branch3x3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels_array[1][0], kernel_size=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels_array[1][0], out_channels_array[1][1], kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True)
        )
        self.branch5x5 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels_array[2][0], kernel_size=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels_array[2][0], out_channels_array[2][1], kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True)
        )
        self.branch_pool = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(in_channels, out_channels_array[3], kernel_size=1),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        return torch.cat([branch1x1, branch3x3, branch5x5, branch_pool], 1)

class InceptionOutClassifier(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # Explicit Kaiming initialization for all Conv2d and Linear layers
        # (done after module construction)
        self.features = torch.nn.Sequential(
            torch.nn.AvgPool2d(kernel_size=5, stride=3), # output 4x4
            torch.nn.Conv2d(in_channels, 128, kernel_size=1),
            torch.nn.Flatten()
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(128 * 4 * 4, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, num_classes),
            #torch.nn.Softmax(dim=1)
        )

        # Explicit Kaiming initialization for all Conv2d and Linear layers
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #print('AuxClassifier input:', x.shape)
        x = self.features(x)
        #print('AuxClassifier after features:', x.shape)
        x = self.classifier(x)
        #print('AuxClassifier after classifier:', x.shape)
        return x
