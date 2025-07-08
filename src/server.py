import cv2
import numpy as np
from ultralytics import YOLO
import os
import gradio as gr
import tempfile

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
        return [], None
    
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
        cv2.putText(img, "识别失败", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)
    
    # 创建临时文件保存结果
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_file.close()
    save_path = temp_file.name
    cv2.imwrite(save_path, img)
    print(f"检测结果已保存至: {save_path}")
    
    return predictions, save_path

def predict_video(video_path):
    """
    对指定路径的视频进行目标检测，并生成带标注的视频
    :param video_path: 视频的路径
    :return: 处理后的视频路径，如果处理失败返回None
    """
    # 加载训练好的YOLO模型
    model = YOLO('/workspaces/yolo--Fruit-Classification/src/runs/detect/train7/weights/best.pt')
    
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法读取视频: {video_path}")
        return None, "无法读取视频"
    
    # 获取视频的帧率和尺寸
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 创建临时文件保存结果视频
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_file.close()
    save_path = temp_file.name
    
    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    
    frame_count = 0
    detected_frames = 0
    
    # 逐帧处理视频
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 进行目标检测
        results = model(frame)
        
        # 检查是否有检测结果
        has_detections = False
        for r in results:
            if len(r.boxes) > 0:
                has_detections = True
                detected_frames += 1
                break
        
        # 遍历检测结果，绘制检测框和标签
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                label = model.names[class_id]
                # 获取检测框坐标并转换为整数
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # 绘制矩形框，颜色为绿色，线条宽度为2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 绘制类别标签，字体为HERSHEY_SIMPLEX，字号0.9，颜色绿色，线条宽度2
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2)
        
        # 如果整帧都没有检测到目标，添加提示文字
        if not has_detections:
            cv2.putText(frame, "识别失败", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
        
        # 写入处理后的帧
        out.write(frame)
        frame_count += 1
    
    # 释放资源
    cap.release()
    out.release()
    
    # 检查是否所有帧都识别失败
    if detected_frames == 0:
        return save_path, "视频中所有帧均识别失败"
    
    return save_path, f"处理完成，共处理 {frame_count} 帧，其中 {detected_frames} 帧有检测结果"

def gradio_predict_image(image):
    """Gradio接口的图片预测函数"""
    if image is None:
        return None, "请上传图片"
    
    # 创建临时文件保存上传的图片
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_file.close()
    image_path = temp_file.name
    
    # 保存图片
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, img)
    
    # 进行预测
    predictions, result_path = predict_image(image_path)
    
    # 读取结果图片
    if result_path:
        result_img = cv2.imread(result_path)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        return result_img, f"识别结果: {', '.join(predictions)}" if predictions else "识别失败"
    else:
        return None, "处理过程中出错"

def gradio_predict_video(video):
    """Gradio接口的视频预测函数"""
    if video is None:
        return None, "请上传视频"
    
    # 进行预测
    result_path, message = predict_video(video)
    
    return result_path, message

def create_web_interface():
    """创建Gradio网页界面"""
    with gr.Blocks(title="水果识别系统") as demo:
        gr.Markdown("# 水果识别系统")
        gr.Markdown("上传图片或视频进行水果识别")
        
        with gr.Tabs():
            with gr.Tab("图片识别"):
                with gr.Row():
                    with gr.Column():
                        img_input = gr.Image(label="上传图片", type="numpy")
                        img_predict_btn = gr.Button("识别水果", variant="primary")
                    
                    with gr.Column():
                        img_output = gr.Image(label="识别结果")
                        img_result_text = gr.Textbox(label="识别结果文本")
                
                img_predict_btn.click(
                    fn=gradio_predict_image,
                    inputs=[img_input],
                    outputs=[img_output, img_result_text]
                )
            
            with gr.Tab("视频识别"):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(label="上传视频")
                        video_predict_btn = gr.Button("识别视频", variant="primary")
                    
                    with gr.Column():
                        video_output = gr.Video(label="识别结果")
                        video_result_text = gr.Textbox(label="处理信息")
                
                video_predict_btn.click(
                    fn=gradio_predict_video,
                    inputs=[video_input],
                    outputs=[video_output, video_result_text]
                )
        
        gr.Markdown("""
        ### 使用说明
        - **图片识别**: 上传图片后点击"识别水果"按钮，系统将在图片上标注出识别到的水果
        - **视频识别**: 上传视频后点击"识别视频"按钮，系统将处理视频并生成标注后的新视频
        
        ### 注意事项
        - 识别结果取决于模型训练的水果类别
        - 视频处理可能需要较长时间，具体取决于视频长度和复杂度
        """)
    
    return demo

if __name__ == '__main__':
    # 创建并启动Gradio界面
    demo = create_web_interface()
    demo.launch()