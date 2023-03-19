from __future__ import print_function
import argparse
import glob
import os
import json
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, help='json file containing annotations')
    parser.add_argument(
        "--output_path", type=str, help='Output directory for image.txt files')
    args = parser.parse_args()

    if args.output_path not in os.listdir():
         os.makedirs(args.output_path)

    with open(args.path) as f:
        data = json.load(f)
    images = data['images']
    annotations = data['annotations']

    file_names = []
    for i in range(0, len(images)):
        file_names.append(images[i]['file_name'])

    cnt = 0
    for i in tqdm(range(0, len(images))):
        converted_results = []
        for ann in annotations:
            if ann['image_id'] == i:
                cat_id = int(ann['category_id'])
                # Yolo classes are starting from zero index
                cat_id -= 1
                h, w, f = images[i]['height'], images[i]['width'], images[i]['file_name']

                box = np.array(ann['bbox'], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                result = (cat_id, box[0], box[1], box[2], box[3])
                
                converted_results.append(result)

        image_name = images[i]['file_name']
        image_name = image_name[5:-4]

        with open(f'{args.output_path}/' + str(image_name) + '.txt', 'w') as file:
            file.write('\n'.join('%d %.6f %.6f %.6f %.6f' % res for res in converted_results))
            file.close()
