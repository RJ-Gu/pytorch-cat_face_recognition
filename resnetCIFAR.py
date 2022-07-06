import torch.nn
import torchvision

resnet = torchvision.models.resnet50()
resnet.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True)

print(resnet)

torch.save(resnet, "resnet.pth")
