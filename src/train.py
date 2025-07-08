from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')

# 开始训练
results = model.train(data='D:\\实习\\fruit_classification\\fruit_classification_project\\src\config\\fruit.yaml', epochs=50, imgsz=640)