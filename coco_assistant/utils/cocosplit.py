"""
Slightly modified version of cocosplit.py by Artur Karaźniewicz (@akarazniewicz)
Link: https://github.com/akarazniewicz/cocosplit/blob/master/cocosplit.py
"""

import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'licenses': licenses, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)

def filter_annotations(annotations, images):
    image_ids = set([x['id'] for x in images])
    return [x for x in tqdm(annotations) if x['image_id'] in image_ids]

def split(ann_path, train_path, test_path, train_size=0.9, random_state=None):
    with open(ann_path, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        info = coco['info']
        licenses = coco['licenses']
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        x, y = train_test_split(images, train_size=train_size, random_state=random_state)

        save_coco(train_path, info, licenses, x, filter_annotations(annotations, x), categories)
        save_coco(test_path, info, licenses, y, filter_annotations(annotations, y), categories)

        print("Saved {} entries in {} and {} in {}".format(len(x), train_path, len(y), test_path))