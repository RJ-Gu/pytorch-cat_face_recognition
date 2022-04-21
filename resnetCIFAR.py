import torch.nn
import torchvision

resnet = torchvision.models.resnet18(pretrained=False)

resnet.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True)

print(resnet)
