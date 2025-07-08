import cv2
import numpy as np
from ultralytics import YOLO
import os


def predict_image(image_path):
    """
    对指定路径的图片进行目标检测，并在图片上绘制检测结果，返回识别出的类别列表。
    :param image_path: 图片的路径
    :return: 识别出的类别列表，如果未识别到则为空列表
    """
    # 加载训练好的YOLO模型
    model = YOLO('/workspaces/yolo--Fruit-Classification/src/runs/detect/train7/weights/best.pt')
    # 读取图片，cv2.imread读取失败会返回None
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}，请检查文件路径和文件完整性。")
        return []
    # 进行目标检测
    results = model(img)
    predictions = []
    # 遍历检测结果，绘制检测框和标签
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            predictions.append(label)
            # 获取检测框坐标并转换为整数
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # 绘制矩形框，颜色为绿色，线条宽度为2
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 绘制类别标签，字体为HERSHEY_SIMPLEX，字号0.9，颜色绿色，线条宽度2
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)
    # 如果没有识别到任何目标，在图片上添加提示文字
    if not predictions:
        cv2.putText(img, "未识别到目标", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)
    # 保存绘制后的图片，覆盖式保存（如果不想覆盖可修改保存路径和文件名逻辑）
    save_path = 'result_image.jpg'
    cv2.imwrite(save_path, img)
    print(f"检测结果已保存至: {os.path.abspath(save_path)}")
    return predictions


if __name__ == '__main__':
    # 图片路径，这里使用os.path.join来拼接路径会更规范，避免不同系统路径分隔符问题
    image_path = os.path.join('..', 'data', 'test', '/workspaces/yolo--Fruit-Classification/data/dataset/李子3.jpg')
    result = predict_image(image_path)
    print("识别结果类别：", result)

    # 读取保存后的结果图片用于显示（因为原始图片可能没做标注，而结果图片是带标注或提示的）
    result_img = cv2.imread('result_image.jpg')
    if result_img is not None:
        cv2.imshow('Prediction Result', result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("显示结果图片失败，可能是保存过程出现问题。")