import os.path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

resnet = torch.load("FaceResNet.pth").cuda()


class TestDataset(Dataset):
    def __init__(self, root_dir, label_dir):  # rootdir: 科大猫咪 label_dir: 全黑
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)  # path: 科大猫咪/全黑
        self.img_path = os.listdir(self.path)

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.path, img_name)
        img = Image.open(img_item_path).convert("RGB")
        img = img.resize((32, 32))
        pil_to_tensor = transforms.PILToTensor()
        img_tensor = pil_to_tensor(img)
        img_tensor = img_tensor.float()
        label = self.label_dir
        return img_tensor, label


train_data_path = "facedata/train"
test_data_path = "facedata/test"
name_list = os.listdir(train_data_path)
label_convert = {}
i = 0
for names in name_list:
    label_convert[names] = i
    i += 1
del i

label_convert_reverse = {}
for names, index in label_convert.items():
    label_convert_reverse[index] = names

root_dir = "facedata"
label = "NewTest"
test_dataset = TestDataset(root_dir, label)

test_dataloader = DataLoader(test_dataset, batch_size=5)

for data in test_dataloader:
    img, label1 = data
    img = img.cuda()
    output = resnet(img)
    return_list = output.argmax(1).cpu().numpy()
    for items in return_list:
        print("{}\n".format(label_convert_reverse[items]))
