import os
import pandas as pd
import torch
import torchvision.models
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# ['三花', '全白', '全黑', '其他', '奶牛', '橘白', '狸花', '玳瑁', '纯橘']
label_convert = {'三花': 1, '全白': 2, '全黑': 3, '其他': 4, '奶牛': 5, '橘白': 6, '狸花': 7, '玳瑁': 8, '纯橘': 0}
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 下面老是报错 shape 不一致


class CatBreed(Dataset):
    def __init__(self, root_dir, label_dir):  # root_dir: 科大猫咪 label_dir: 全黑
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
        label = label_convert[self.label_dir]
        return img_tensor, label


root_dir = "科大猫咪/train"
cat_labels = os.listdir(root_dir)  # cat_labels: [全黑, 全白, 玳瑁, ...](list)
flag = 0
for items in cat_labels:
    if flag == 0:
        train_data = CatBreed(root_dir, items)
        flag = 1
    else:
        train_data = train_data + CatBreed(root_dir, items)

root_dir = "科大猫咪/test"
cat_labels = os.listdir(root_dir)  # cat_labels: [全黑, 全白, 玳瑁, ...](list)
flag = 0
for items in cat_labels:
    if flag == 0:
        test_data = CatBreed(root_dir, items)
        flag = 1
    else:
        test_data = test_data + CatBreed(root_dir, items)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集长度为：{}".format(train_data_size))
print("测试数据集长度为：{}".format(test_data_size))

resnet = torchvision.models.resnet18(pretrained=False)
resnet.fc = torch.nn.Linear(in_features=512, out_features=9, bias=True)
resnet = resnet.cuda()

train_dataloader = DataLoader(train_data, batch_size=5, shuffle=True, drop_last=False)
test_dataloader = DataLoader(test_data, batch_size=9)

loss_fn = torch.nn.CrossEntropyLoss().cuda()

learning_rate = 1.5e-2
optimizer = torch.optim.SGD(resnet.parameters(), lr=learning_rate)

total_train_step = 0
total_test_step = 0
epoch = 50

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
        if total_train_step % 10 == 0:
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
    excel_writer=r'train_data.xlsx',
    sheet_name='train',
    index=False,
    columns=["次数", "损失率"],
    encoding="GBK")
test_excel.to_excel(
    excel_writer=r'test_data.xlsx',
    sheet_name='test',
    index=False,
    columns=["次数", "准确率"],
    encoding="GBK")
