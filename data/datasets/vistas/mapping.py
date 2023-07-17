import json
import os
import torch
from data.datasets.cityscapes.cityscapes_labels import create_name_to_id, create_id_to_name

_vistas_to_cityscapes = {
    'construction--barrier--curb': 'sidewalk',
    'construction--barrier--fence': 'fence',
    'construction--barrier--guard-rail': 'fence',
    'construction--barrier--wall': 'wall',
    'construction--flat--bike-lane': 'road',
    'construction--flat--crosswalk-plain': 'road',
    'construction--flat--curb-cut': 'sidewalk',
    'construction--flat--parking': 'road',
    'construction--flat--pedestrian-area': 'sidewalk',
    'construction--flat--rail-track': 'road',
    'construction--flat--road': 'road',
    'construction--flat--service-lane': 'road',
    'construction--flat--sidewalk': 'sidewalk',
    'construction--structure--bridge': 'building',
    'construction--structure--building': 'building',
    'construction--structure--tunnel': 'building',
    'human--person': 'person',
    'human--rider--bicyclist': 'rider',
    'human--rider--motorcyclist': 'rider',
    'human--rider--other-rider': 'rider',
    'marking--crosswalk-zebra': 'road',
    'marking--general': 'road',
    'nature--sand': 'terrain',
    'nature--sky': 'sky',
    'nature--snow': 'terrain',
    'nature--terrain': 'terrain',
    'nature--vegetation': 'vegetation',
    'object--support--pole': 'pole',
    'object--support--traffic-sign-frame': 'traffic sign',
    'object--support--utility-pole': 'pole',
    'object--traffic-light': 'traffic light',
    'object--traffic-sign--front': 'traffic sign',
    'object--vehicle--bicycle': 'bicycle',
    'object--vehicle--bus': 'bus',
    'object--vehicle--car': 'car',
    'object--vehicle--motorcycle': 'motorcycle',
    'object--vehicle--on-rails': 'train',
    'object--vehicle--truck': 'truck',
}

cityscapes_name_to_id = create_name_to_id()
cityscapes_id_to_name = create_id_to_name()
cs_ignore_class = 255
n_classes = 66


def _parse_config(config_path):
    # read in config file
    with open(config_path) as config_file:
        config = json.load(config_file)

    labels = config['labels']

    class_names = []
    class_ids = []
    class_colors = []
    id_to_name = {}
    # print("> There are {} labels in the config file".format(len(labels)))
    for label_id, label in enumerate(labels):
        class_names.append(label["readable"])
        class_ids.append(label_id)
        class_colors.append(label["color"])
        id_to_name[label_id] = label["name"]
    return class_names, class_ids, class_colors, id_to_name, labels


def _to_cityscapes_class(id, vistas_id_to_name):
    vistas_name = vistas_id_to_name[id]
    cityscapes_name = _vistas_to_cityscapes.get(vistas_name)
    if cityscapes_name == None:
        return cityscapes_name_to_id['ignore']
    else:
        return cityscapes_name_to_id[cityscapes_name]

def create_vistas_to_cityscapes_mapper(root):
    class_names, class_ids, class_colors, vistas_id_to_name, labels = _parse_config(os.path.join(root, 'config.json'))
    cityscapes_classes_mapper = torch.zeros(n_classes).long().fill_(cs_ignore_class)
    for i in range(len(labels)):
        cityscapes_classes_mapper[i] = _to_cityscapes_class(i, vistas_id_to_name)
    return cityscapes_classes_mapper