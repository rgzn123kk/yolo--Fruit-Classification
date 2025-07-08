def predict_image(model, image_path, save_path):
    img = cv2.imread(image_path)
    results = model(img)
    # ...处理结果并保存图片到 save_path
    cv2.imwrite(save_path, img)
