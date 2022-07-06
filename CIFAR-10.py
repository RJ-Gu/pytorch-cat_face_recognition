import torch
from torch import nn
import torchvision.datasets
from torch.utils.data import DataLoader
import pandas as pd

train_data = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集长度为：{}".format(train_data_size))
print("测试数据集长度为：{}".format(test_data_size))

resnet = torch.load("resnet.pth").cuda()

train_dataloader = DataLoader(train_data, batch_size=100, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=100)

loss_fn = nn.CrossEntropyLoss().cuda()

learning_rate = 1e-2
optimizer = torch.optim.SGD(resnet.parameters(), lr=learning_rate)

total_train_step = 0
total_test_step = 0
epoch = 10

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
    test_excel['次数'].append(format(total_test_step))
    test_excel['准确率'].append(format(total_accuracy/test_data_size))
    total_test_step += 1

train_excel = pd.DataFrame(train_excel)
test_excel = pd.DataFrame(test_excel)
train_excel.to_excel(
    excel_writer=r'CIFAR_train_data.xlsx',
    sheet_name='train',
    index=False,
    columns=["次数", "损失率"],
    encoding="GBK")
test_excel.to_excel(
    excel_writer=r'CIFAR_test_data.xlsx',
    sheet_name='test',
    index=False,
    columns=["次数", "准确率"],
    encoding="GBK")
