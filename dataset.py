from torch.utils.data import Dataset
import os
from PIL import Image


class CatBreed(Dataset):
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
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label


root_dir = "科大猫咪"
cat_labels = os.listdir(root_dir)  # cat_labels: [全黑, 全白, 玳瑁, ...](list)
flag = 0
for items in cat_labels:
    if flag == 0:
        cat_dataset = CatBreed(root_dir, items)
        flag = 1
    else:
        cat_dataset = cat_dataset + CatBreed(root_dir, items)
