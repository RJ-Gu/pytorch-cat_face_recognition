import os
import sys
import pandas as pd
import torch
import torchvision.models
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


class ManName(Dataset):
    def __init__(self, data_path, label_path):  # data_path: facedata; label_path: aaron
        self.data_path = data_path
        self.label_path = label_path
        self.path = os.path.join(self.data_path, self.label_path)  # self.path:facedata\aaron
        self.video_list = os.listdir(self.path)  # [0, 1, 2]
        self.img_num_list = []
        for videos in self.video_list:
            video_names = os.path.join(self.path, videos)  # video_names: facedata\aaron\0
            img_list = os.listdir(video_names)  # [img1, img2]
            self.img_num_list.append(len(img_list))

    def __len__(self):
        img_len = 0
        for videos in self.video_list:
            video_dir = os.path.join(self.path, videos)
            img_list = os.listdir(video_dir)
            img_len = img_len + len(img_list)
        return img_len

    def __getitem__(self, index):
        video_category = 0
        for i in range(len(self.img_num_list)):
            if index < self.img_num_list[i]:
                video_category = i
                break
            index = index - self.img_num_list[i]
        videos = self.video_list[video_category]
        video_names = os.path.join(self.path, videos)  # video_names: facedata\aaron\0
        img_list = os.listdir(video_names)  # [img1, img2]
        img_path = os.path.join(videos, img_list[index])  # img_path: 0\img1
        img = Image.open(os.path.join(self.path, img_path)).convert("RGB")
        img = img.resize((32, 32))
        pil_to_tensor = transforms.PILToTensor()
        img_tensor = pil_to_tensor(img)
        img_tensor = img_tensor.float()
        label = label_convert[self.label_path]
        return img_tensor, label


sys.setrecursionlimit(100000)

train_data_path = "facedata/train"
test_data_path = "facedata/test"
name_list = os.listdir(train_data_path)
label_convert = {}
i = 0
for names in name_list:
    label_convert[names] = i
    i += 1
del i

flag = 0
for items in name_list:
    if flag == 0:
        train_data = ManName(train_data_path, items)
        flag = 1
    else:
        train_data = train_data + ManName(train_data_path, items)

flag = 0
for items in name_list:
    if flag == 0:
        test_data = ManName(test_data_path, items)
        flag = 1
    else:
        test_data = test_data + ManName(test_data_path, items)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集长度为：{}".format(train_data_size))
print("测试数据集长度为：{}".format(test_data_size))

resnet = torchvision.models.resnet18(pretrained=False)
resnet.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(3, 3))
resnet.fc = torch.nn.Linear(in_features=4608, out_features=1600, bias=True)
resnet = resnet.cuda()

train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True, drop_last=False)
test_dataloader = DataLoader(test_data, batch_size=128, shuffle=True)

loss_fn = torch.nn.CrossEntropyLoss().cuda()

learning_rate = 1e-2
optimizer = torch.optim.SGD(resnet.parameters(), lr=learning_rate)

total_train_step = 0
total_test_step = 0
epoch = 3

writer = SummaryWriter("./logs")
train_excel = {'次数': [], '损失率': []}
test_excel = {'次数': [], '准确率': []}

for i in range(epoch):
    print("第{}轮训练开始".format(i + 1))
    for data in train_dataloader:
        images, targets = data
        images = images.cuda()
        targets = targets.cuda()
        outputs = resnet(images)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，Loss：{}".format(total_train_step, loss))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
            train_excel['次数'].append(format(total_train_step))
            train_excel['损失率'].append(format(loss))

    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            images, targets = data
            images = images.cuda()
            targets = targets.cuda()
            outputs = resnet(images)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    test_excel['次数'].append(format(total_test_step))
    test_excel['准确率'].append(format(total_accuracy / test_data_size))
    total_test_step += 1

writer.close()
train_excel = pd.DataFrame(train_excel)
test_excel = pd.DataFrame(test_excel)
train_excel.to_excel(
    excel_writer=r'Face_train_data.xlsx',
    sheet_name='train',
    index=False,
    columns=["次数", "损失率"],
    encoding="GBK")
test_excel.to_excel(
    excel_writer=r'Face_test_data.xlsx',
    sheet_name='test',
    index=False,
    columns=["次数", "准确率"],
    encoding="GBK")

torch.save(resnet, "FaceResNet.pth")
