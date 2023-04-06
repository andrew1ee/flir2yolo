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
        # 10: 7,  # light
        # 11: 8,  # hydrant
        12: 7,  # sign
        17: 8, # dog
        18: 9, # deer
        37: 10, # skateboard
        73: 11, # stroller
        75: 12, # scooter
        79: 13  # other vehicle
    }
    flir = CAT_MAP[coco] 
    return flir

if __name__ == '__main__':
    base_folder = '../FLIR_ADAS_v2_converted'
    folders = ['images_rgb_train', 'images_rgb_val', 'images_thermal_train', 'images_thermal_val', 'video_rgb_test', 'video_thermal_test']

    omit_categories = [10, 11]

    for name in folders:
        input_json = f'{base_folder}/{name}/coco.json'
        output_path = f'{base_folder}/{name}/labels'
        os.mkdir(output_path)
        print(f'Converting {name}')

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
                if ann['image_id'] == id and int(ann['category_id']) not in omit_categories:
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
