from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

train_data = torchvision.datasets.CIFAR10(root="./data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)

writer = SummaryWriter("logs")

for i in range(10):
    img, target = train_data[i]
    writer.add_image("123", img, i)
writer.close()
