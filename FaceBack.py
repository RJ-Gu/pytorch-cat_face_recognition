import os
import shutil

rootdir = "facedata/Bilibili up add"
train_dir = os.path.join(rootdir, "train")  # facedata/train
test_dir = os.path.join(rootdir, "test")  # facedata/test

train_list = os.listdir(train_dir)  # [aaron, abba, ...]
for names in train_list:  # aaron
    train_person_name_dir = os.path.join(train_dir, names)  # facedata/train/aaron
    test_person_name_dir = os.path.join(test_dir, names)  # facedata/test/aaron
    if not os.path.isdir(test_person_name_dir):
        os.mkdir(test_person_name_dir)
    videos_list = os.listdir(train_person_name_dir)
    for i in range(len(videos_list)):
        train_video_dir = os.path.join(train_person_name_dir, videos_list[i])  # facedata/train/aaron/0
        test_video_dir = os.path.join(test_person_name_dir, videos_list[i])  # facedata/test/aaron/0
        if not os.path.isdir(test_video_dir):
            os.mkdir(test_video_dir)
        test_img_list = os.listdir(test_video_dir)  # [img1,img2,img3]
        for j in range(len(test_img_list)):
            src_img_path = os.path.join(test_video_dir, test_img_list[j])
            shutil.move(src_img_path, train_video_dir)
