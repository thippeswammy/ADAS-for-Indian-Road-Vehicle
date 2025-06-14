import os
import json

import cv2
import yaml

train_path = '../road/train'
val_path = '../road/valid'
test_path = '../road/test'

source_path_ = '../idd20kII/Img/'

# Initialize an empty dictionary for class name to ID mapping
class_name_to_id = {}
current_class_id = 0


def move_files():
    count = 1
    for root, dirs, files in os.walk(source_path_):
        for filename in files:
            if filename.endswith('.jpg'):
                file_path = os.path.join(root, filename)
                save(file_path, count)
                count += 1


def save(source_path_, count):
    source_path_ = os.path.normpath(source_path_)
    source_path_json = source_path_.split(os.sep)
    localPath = val_path
    if source_path_json[3] == 'train':
        localPath = train_path
    elif source_path_json[3] == 'test':
        localPath = test_path
    source_path_json[2] = 'Seg'
    source_path_json = '/'.join(source_path_json)[:-16] + '_gtFine_polygons.json'

    destination_path_img = os.path.join(localPath, 'images', f'File{count}.jpg')
    destination_path_lab = os.path.join(localPath, 'labels', f'File{count}.txt')

    if os.path.exists(source_path_json):
        cv2.imwrite(destination_path_img, cv2.imread(source_path_))
        convert_json_to_txt(source_path_json, destination_path_lab)


def convert_json_to_txt(json_file, output_txt_file):
    global current_class_id

    with open(json_file, 'r') as f:
        data = json.load(f)

    img_width = data['imgWidth']
    img_height = data['imgHeight']

    with open(output_txt_file, 'a') as out_file:
        for obj in data['objects']:
            class_name = obj['label']

            # Normalize the polygon coordinates
            normalized_polygon = [
                (point[0] / img_width, point[1] / img_height)
                for point in obj['polygon']
            ]
            if class_name not in class_name_to_id:
                class_name_to_id[class_name] = current_class_id
                current_class_id += 1
            # Convert normalized polygon to string
            polygon_str = ' '.join(f"{x:.6f} {y:.6f}" for x, y in normalized_polygon)
            class_id = class_name_to_id[class_name]
            # Write to file in YOLO format: <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
            out_file.write(f"{class_id} {polygon_str}\n")


move_files()
# After processing all files, print the class name to ID mapping
print("Class Name to ID Mapping:")
id_to_name = {v: k for k, v in class_name_to_id.items()}
# Convert to YAML format
yaml_data = {'names': id_to_name}
# Print the YAML
print(yaml.dump(yaml_data, default_flow_style=False, sort_keys=False))
