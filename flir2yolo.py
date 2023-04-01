from __future__ import print_function
import argparse
import pybboxes as pbx
import glob
import os
import json
import numpy as np
from tqdm import tqdm

def convert_cat_id(coco):
    CAT_MAP = {
        1: 0,   # person
        2: 1,   # bike
        3: 2,   # car
        4: 3,   # motor
        6: 4,   # bus
        7: 5,   # train
        8: 6,   # truck
        10: 7,  # light
        11: 8,  # hydrant
        12: 9,  # sign
        17: 10, # dog
        18: 11, # deer
        37: 12, # skateboard
        73: 13, # stroller
        75: 14, # scooter
        79: 15  # other vehicle
    }
    flir = CAT_MAP[coco] 
    return flir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, help='json file containing annotations')
    parser.add_argument(
        "--output_path", type=str, help='Output directory for image.txt files')
    args = parser.parse_args()

    folders = ['train_rgb', 'train_thermal', 'valid_rgb', 'valid_thermal', 'test_rgb', 'test_thermal']
    for name in folders:
        input_json = f'cocos/{name}_coco.json'
        output_path = f'{name}/labels'
        os.mkdir(output_path)

        with open(input_json) as f:
            data = json.load(f)
        images = data['images']
        annotations = data['annotations']

        file_names = []
        for i in range(0, len(images)):
            file_names.append(images[i]['file_name'])

        cnt = 0
        image_id_start = annotations[0]["image_id"]
        image_id_end = annotations[-1]["image_id"]
        for i in tqdm(range(0, len(images))):
            converted_results = []
            id = images[i]['id']
            for ann in annotations:
                if ann['image_id'] == id:
                    cat_id = convert_cat_id(int(ann['category_id']))
                    h, w, f = images[i]['height'], images[i]['width'], images[i]['file_name']

                    coco_box = np.array(ann['bbox'], dtype=np.float64)
                    box = pbx.convert_bbox(coco_box, from_type="coco", to_type="yolo", image_size=(w, h))

                    result = (cat_id, box[0], box[1], box[2], box[3])
                    converted_results.append(result)

            image_name = f[5:-4]

            with open(f'{output_path}/' + str(image_name) + '.txt', 'w') as file:
                file.write('\n'.join('%d %.6f %.6f %.6f %.6f' % res for res in converted_results))
                file.close()
