import importlib.util
import subprocess
import sys

def check_and_install(package_name, import_name=None):
    """检查包是否已安装，如果未安装则使用pip进行安装"""
    # 如果没有提供导入名称，则使用包名作为导入名称
    if import_name is None:
        import_name = package_name
    
    # 检查包是否已安装
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        print(f"未找到 {package_name}，正在安装...")
        try:
            # 使用pip安装包
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package_name
            ])
            print(f"{package_name} 安装成功")
        except subprocess.CalledProcessError as e:
            print(f"安装 {package_name} 失败: {e}")
            sys.exit(1)
    else:
        print(f"{package_name} 已安装")

def main():
    # 定义需要检查和安装的包
    packages_to_check = [
        ("torch", "torch"),           # PyTorch
        ("gradio", "gradio"),         # Gradio
        ("ultralytics", "ultralytics"), # Ultralytics (YOLOv8)
        ("Pillow", "PIL"),            # Pillow (PIL)
        ("opencv-python", "cv2")      # OpenCV
    ]
    
    print("开始检查依赖包...")
    
    # 检查并安装每个包
    for package_name, import_name in packages_to_check:
        check_and_install(package_name, import_name)
    
    print("所有依赖包检查完毕！")

if __name__ == "__main__":
    main()