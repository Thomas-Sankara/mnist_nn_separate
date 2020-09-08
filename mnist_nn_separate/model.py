from torch import nn
import torch.nn.functional as F

# 我们将使用 Pytorch 的预定义 Conv2d 类作为我们的卷积层。 我们定义具有 3 个卷积层的 CNN。
# 每个卷积后跟一个 ReLU。 最后，我们执行平均池化。 (请注意，view是 numpy 的reshape的 PyTorch 版本）
class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))
