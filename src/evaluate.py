from ultralytics import YOLO


def evaluate_model():
    model = YOLO('./runs/detect/train7/weights/best.pt')
    results = model.val(data='D:\\实习\\fruit_classification\\fruit_classification_project\\src\\config\\fruit.yaml')
    print(f"mAP@0.5: {results.box.map50}")
    print(f"mAP@0.5:0.95: {results.box.map}")


if __name__ == '__main__':
    evaluate_model()