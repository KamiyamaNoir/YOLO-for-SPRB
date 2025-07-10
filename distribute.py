"""
VoTT导出为csv格式
用于将VoTT工程转化为YOLO可识别的训练集
将vott-csv-export置于该目录下，修改csv文件名称，运行脚本，得到output
需要根据实际情况修改output为train或val或test，再编写dataset.yaml
"""
import os
import csv
from PIL import Image

# 标签映射表
tags2number = {
    'ball': 0,
    'target': 1,
    'rindan': 2
}

os.makedirs('./output/images', exist_ok=True)
os.makedirs('./output/labels', exist_ok=True)

data_by_image = {}

with open('./vott-csv-export/...', 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        image_name = row['image']
        if image_name not in data_by_image:
            data_by_image[image_name] = []
        data_by_image[image_name].append(row)

for image_name, rows in data_by_image.items():
    source_image_path = os.path.join('./vott-csv-export', image_name)
    target_image_path = os.path.join('./output/images', image_name)
    label_file_name = os.path.splitext(image_name)[0] + '.txt'
    label_file_path = os.path.join('./output/labels', label_file_name)

    if not os.path.exists(target_image_path):
        os.system(f'copy "{source_image_path}" "{target_image_path}"')

    with Image.open(source_image_path) as img:
        img_width, img_height = img.size

    with open(label_file_path, 'w', encoding='utf-8') as label_file:
        for row in rows:
            # 解析坐标和标签
            xmin = float(row['xmin'])
            ymin = float(row['ymin'])
            xmax = float(row['xmax'])
            ymax = float(row['ymax'])
            label = row['label']

            # 计算 YOLO 格式的坐标
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            box_width = xmax - xmin
            box_height = ymax - ymin

            # 归一化坐标
            x_center_norm = x_center / img_width
            y_center_norm = y_center / img_height
            width_norm = box_width / img_width
            height_norm = box_height / img_height

            # 写入标签文件
            label_file.write(f"{tags2number[label]} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")
