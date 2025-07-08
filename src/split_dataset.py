import os
import shutil
from sklearn.model_selection import train_test_split

# 数据集所在目录，包含图像和对应的 YOLO 格式标注文件
data_dir = '../data/dataset'
# 划分后训练集存放目录
train_dir = '../data/train'
# 划分后验证集存放目录
val_dir = '../data/val'
# 划分后测试集存放目录
test_dir = '../data/test'

# 确保输出目录存在，不存在则创建
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 获取所有图像文件
image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg') or f.endswith('.png')]

# 先划分出测试集，test_size 表示测试集占总数据集的比例，这里设为 0.2，即 20%
train_val_images, test_images = train_test_split(image_files, test_size=0.2, random_state=42)

# 再从训练验证集中划分出验证集，test_size 设为 0.25，即训练验证集中 25% 作为验证集
train_images, val_images = train_test_split(train_val_images, test_size=0.25, random_state=42)

# 定义一个函数来移动文件
def move_files(image_list, source_dir, target_dir):
    for img in image_list:
        img_path = os.path.join(source_dir, img)
        shutil.copy(img_path, target_dir)
        # 假设标注文件和图像文件名除了后缀都相同
        label_file = img.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(source_dir, label_file)
        shutil.copy(label_path, target_dir)

# 移动训练集文件
move_files(train_images, data_dir, train_dir)
# 移动验证集文件
move_files(val_images, data_dir, val_dir)
# 移动测试集文件
move_files(test_images, data_dir, test_dir)