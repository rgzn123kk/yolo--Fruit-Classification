import os
import json


def polygon_to_yolo_bbox(points, image_width, image_height):
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    w = x_max - x_min
    h = y_max - y_min
    x_center = (x_min + x_max) / 2 / image_width
    y_center = (y_min + y_max) / 2 / image_height
    norm_width = w / image_width
    norm_height = h / image_height
    return x_center, y_center, norm_width, norm_height


def convert_json_to_yolo(json_file_path, output_dir, class_names):
    with open(json_file_path, 'r', encoding='utf - 8') as f:
        data = json.load(f)
    image_width = data['imageWidth']
    image_height = data['imageHeight']
    output_txt_path = os.path.join(output_dir, os.path.splitext(os.path.basename(json_file_path))[0] + '.txt')
    with open(output_txt_path, 'w', encoding='utf - 8') as f:
        for shape in data['shapes']:
            label = shape['label']
            if label not in class_names:
                continue
            class_index = class_names.index(label)
            points = shape['points']
            x_center, y_center, norm_width, norm_height = polygon_to_yolo_bbox(points, image_width, image_height)
            line = f"{class_index} {x_center} {y_center} {norm_width} {norm_height}\n"
            f.write(line)


def main():
    input_dir = '../data/dataset'
    output_dir = '../data/dataset'
    class_names = ['ping guo', 'li zi', 'xiang jiao', 'tao zi', 'yang tao']
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            json_file_path = os.path.join(input_dir, filename)
            convert_json_to_yolo(json_file_path, output_dir, class_names)


if __name__ == "__main__":
    main()