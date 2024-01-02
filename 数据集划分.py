# import os
# import random
# import shutil
#
# def split_dataset(source_dir, train_dir, test_dir, train_ratio=0.8):
#     # 创建训练集和测试集目录
#     os.makedirs(train_dir, exist_ok=True)
#     os.makedirs(test_dir, exist_ok=True)
#
#     # 获取所有图片文件
#     all_images = [f for f in os.listdir(source_dir) if f.endswith('.jpg') or f.endswith('.png')]
#
#     # 随机打乱顺序
#     random.shuffle(all_images)
#
#     # 计算训练集和测试集的划分索引
#     num_train = int(len(all_images) * train_ratio)
#     train_images = all_images[:num_train]
#     test_images = all_images[num_train:]
#
#     # 将图片移动到对应目录
#     for img in train_images:
#         src_path = os.path.join(source_dir, img)
#         dst_path = os.path.join(train_dir, img)
#         shutil.copy(src_path, dst_path)
#
#     for img in test_images:
#         src_path = os.path.join(source_dir, img)
#         dst_path = os.path.join(test_dir, img)
#         shutil.copy(src_path, dst_path)
#
# # 示例用法
# source_directory = r'C:\Users\H2250\Desktop\vgg16\img\猫咪'
# train_directory = r'C:\Users\H2250\Desktop\vgg16\img\train\cat_1'
# test_directory = r'C:\Users\H2250\Desktop\vgg16\img\val\cat_1'
# split_ratio = 0.7  # 训练集的比例，可以根据需要调整
#
# split_dataset(source_directory, train_directory, test_directory, split_ratio)

