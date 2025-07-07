import os
import subprocess
import sys
from pathlib import Path

def install_dependencies():
    """安装 YOLO8 所需的依赖"""
    try:
        print("正在安装 YOLO8 依赖...")
        subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics", "torch", "torchvision"], check=True)
        print("依赖安装完成。")
    except subprocess.CalledProcessError as e:
        print(f"依赖安装失败: {e}")
        sys.exit(1)

def train_yolo8(data_yaml_path, model_type='yolov8n.pt', epochs=100, imgsz=640):
    """
    使用自定义数据集训练 YOLO8 模型
    
    参数:
        data_yaml_path (str): 数据集配置文件(.yaml)的路径
        model_type (str): 要使用的 YOLO8 模型类型，默认为 'yolov8n.pt' (YOLOv8 Nano)
        epochs (int): 训练轮数，默认为 100
        imgsz (int): 输入图像尺寸，默认为 640
    """
    try:
        from ultralytics import YOLO
        
        # 检查数据集配置文件是否存在
        if not os.path.exists(data_yaml_path):
            print(f"错误: 数据集配置文件 '{data_yaml_path}' 不存在。")
            return
        
        print(f"开始使用 {model_type} 模型训练 YOLO8...")
        
        # 加载模型
        model = YOLO(model_type)
        
        # 训练模型
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            device=0  # 使用 GPU 训练，如果想使用 CPU，设置为 'cpu'
        )
        
        print(f"训练完成。模型权重已保存到 runs/detect/train/weights/best.pt")
        return results
        
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        return None

if __name__ == "__main__":
    # 安装依赖
    install_dependencies()
    
    # 示例：使用自定义数据集训练 YOLO8
    # 请将 'path/to/your/data.yaml' 替换为你实际的数据集配置文件路径
    # 你也可以修改 model_type、epochs 和 imgsz 参数
    data_config_path = 'path/to/your/data.yaml'
    
    print("\n警告: 请先将 'data_config_path' 修改为你实际的数据集配置文件路径!")
    print("示例数据集配置文件格式:")
    print("""
    train: ../train/images
    val: ../valid/images
    
    nc: 2  # 类别数量
    names: ['cat', 'dog']  # 类别名称
    """)
    
    if data_config_path == 'path/to/your/data.yaml':
        print("\n由于路径未修改，训练未执行。请编辑此脚本并设置正确的数据集路径。")
    else:
        train_yolo8(data_config_path, model_type='yolov8n.pt', epochs=50, imgsz=640)    