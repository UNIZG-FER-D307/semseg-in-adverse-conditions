import torch
import numpy as np

wilddash_config = [
        {
            "color": [
                0,
                0,
                0
            ],
            "instances": False,
            "readable": "unlabeled(void)",
            "name": "unlabeled",
            "evaluate": False,
            "id": 0,
            "supercategory": "void"
        },
        {
            "color": [
                0,
                20,
                50
            ],
            "instances": True,
            "readable": "ego vehicle(vehicle)",
            "name": "ego vehicle",
            "evaluate": True,
            "id": 1,
            "supercategory": "vehicle"
        },
        {
            "color": [
                20,
                20,
                20
            ],
            "instances": False,
            "readable": "rectification border(void); this label is also used for image overlays (GPS; timestamps; logos added during postprocesseing; etc.)",
            "name": "rectification border",
            "evaluate": False,
            "id": 2,
            "supercategory": "void"
        },
        {
            "color": [
                0,
                0,
                0
            ],
            "instances": False,
            "readable": "out of roi(void)",
            "name": "out of roi",
            "evaluate": False,
            "id": 3,
            "supercategory": "void"
        },
        {
            "color": [
                0,
                0,
                0
            ],
            "instances": False,
            "readable": "static(void)",
            "name": "static",
            "evaluate": False,
            "id": 4,
            "supercategory": "void"
        },
        {
            "color": [
                111,
                74,
                0
            ],
            "instances": False,
            "readable": "dynamic(void)",
            "name": "dynamic",
            "evaluate": False,
            "id": 5,
            "supercategory": "void"
        },
        {
            "color": [
                81,
                0,
                81
            ],
            "instances": False,
            "readable": "ground(flat)",
            "name": "ground",
            "evaluate": False,
            "id": 6,
            "supercategory": "flat"
        },
        {
            "color": [
                128,
                64,
                128
            ],
            "instances": False,
            "readable": "road(flat)",
            "name": "road",
            "evaluate": True,
            "id": 7,
            "supercategory": "flat"
        },
        {
            "color": [
                244,
                35,
                232
            ],
            "instances": False,
            "readable": "sidewalk(flat)",
            "name": "sidewalk",
            "evaluate": True,
            "id": 8,
            "supercategory": "flat"
        },
        {
            "color": [
                250,
                170,
                160
            ],
            "instances": False,
            "readable": "parking(flat)",
            "name": "parking",
            "evaluate": False,
            "id": 9,
            "supercategory": "flat"
        },
        {
            "color": [
                230,
                150,
                140
            ],
            "instances": False,
            "readable": "rail track(flat)",
            "name": "rail track",
            "evaluate": False,
            "id": 10,
            "supercategory": "flat"
        },
        {
            "color": [
                70,
                70,
                70
            ],
            "instances": False,
            "readable": "building(construction)",
            "name": "building",
            "evaluate": True,
            "id": 11,
            "supercategory": "construction"
        },
        {
            "color": [
                102,
                102,
                156
            ],
            "instances": False,
            "readable": "wall(construction)",
            "name": "wall",
            "evaluate": True,
            "id": 12,
            "supercategory": "construction"
        },
        {
            "color": [
                190,
                153,
                153
            ],
            "instances": False,
            "readable": "fence(construction)",
            "name": "fence",
            "evaluate": True,
            "id": 13,
            "supercategory": "construction"
        },
        {
            "color": [
                180,
                165,
                180
            ],
            "instances": False,
            "readable": "guard rail(construction)",
            "name": "guard rail",
            "evaluate": True,
            "id": 14,
            "supercategory": "construction"
        },
        {
            "color": [
                150,
                100,
                100
            ],
            "instances": False,
            "readable": "bridge(construction)",
            "name": "bridge",
            "evaluate": False,
            "id": 15,
            "supercategory": "construction"
        },
        {
            "color": [
                150,
                120,
                90
            ],
            "instances": False,
            "readable": "tunnel(construction)",
            "name": "tunnel",
            "evaluate": False,
            "id": 16,
            "supercategory": "construction"
        },
        {
            "color": [
                153,
                153,
                153
            ],
            "instances": False,
            "readable": "pole(object)",
            "name": "pole",
            "evaluate": True,
            "id": 17,
            "supercategory": "object"
        },
        {
            "color": [
                153,
                153,
                153
            ],
            "instances": False,
            "readable": "polegroup(object); this labelid is not used by WildDash but remains in the list to preserve Cityscapes compatibility",
            "name": "polegroup",
            "evaluate": False,
            "id": 18,
            "supercategory": "object"
        },
        {
            "color": [
                250,
                170,
                30
            ],
            "instances": False,
            "readable": "traffic light(object)",
            "name": "traffic light",
            "evaluate": True,
            "id": 19,
            "supercategory": "object"
        },
        {
            "color": [
                220,
                220,
                0
            ],
            "instances": False,
            "readable": "traffic sign(object)",
            "name": "traffic sign",
            "evaluate": True,
            "id": 20,
            "supercategory": "object"
        },
        {
            "color": [
                107,
                142,
                35
            ],
            "instances": False,
            "readable": "vegetation(nature)",
            "name": "vegetation",
            "evaluate": True,
            "id": 21,
            "supercategory": "nature"
        },
        {
            "color": [
                152,
                251,
                152
            ],
            "instances": False,
            "readable": "terrain(nature)",
            "name": "terrain",
            "evaluate": True,
            "id": 22,
            "supercategory": "nature"
        },
        {
            "color": [
                70,
                130,
                180
            ],
            "instances": False,
            "readable": "sky(sky)",
            "name": "sky",
            "evaluate": True,
            "id": 23,
            "supercategory": "sky"
        },
        {
            "color": [
                220,
                20,
                60
            ],
            "instances": True,
            "readable": "person(human)",
            "name": "person",
            "evaluate": True,
            "id": 24,
            "supercategory": "human"
        },
        {
            "color": [
                255,
                0,
                0
            ],
            "instances": True,
            "readable": "rider(human)",
            "name": "rider",
            "evaluate": True,
            "id": 25,
            "supercategory": "human"
        },
        {
            "color": [
                0,
                0,
                142
            ],
            "instances": True,
            "readable": "car(vehicle)",
            "name": "car",
            "evaluate": True,
            "id": 26,
            "supercategory": "vehicle"
        },
        {
            "color": [
                0,
                0,
                70
            ],
            "instances": True,
            "readable": "truck(vehicle)",
            "name": "truck",
            "evaluate": True,
            "id": 27,
            "supercategory": "vehicle"
        },
        {
            "color": [
                0,
                60,
                100
            ],
            "instances": True,
            "readable": "bus(vehicle)",
            "name": "bus",
            "evaluate": True,
            "id": 28,
            "supercategory": "vehicle"
        },
        {
            "color": [
                0,
                0,
                90
            ],
            "instances": True,
            "readable": "caravan(vehicle)",
            "name": "caravan",
            "evaluate": False,
            "id": 29,
            "supercategory": "vehicle"
        },
        {
            "color": [
                0,
                0,
                110
            ],
            "instances": True,
            "readable": "trailer(vehicle)",
            "name": "trailer",
            "evaluate": False,
            "id": 30,
            "supercategory": "vehicle"
        },
        {
            "color": [
                0,
                80,
                100
            ],
            "instances": True,
            "readable": "trains and trams, original cityscapes name is 'on rails' (vehicle)",
            "name": "train",
            "evaluate": False,
            "id": 31,
            "supercategory": "vehicle"
        },
        {
            "color": [
                0,
                0,
                230
            ],
            "instances": True,
            "readable": "motorcycle (vehicle)",
            "name": "motorcycle",
            "evaluate": True,
            "id": 32,
            "supercategory": "vehicle"
        },
        {
            "color": [
                119,
                11,
                32
            ],
            "instances": True,
            "readable": "bicycle (vehicle)",
            "name": "bicycle",
            "evaluate": True,
            "id": 33,
            "supercategory": "vehicle"
        },
        {
            "color": [
                40,
                0,
                100
            ],
            "instances": True,
            "readable": "pickup-truck (vehicle); with a car-like front and an open cargo area which is not a fully seperated part of the vehicle but integrated into the vehicle's frame",
            "name": "pickup",
            "evaluate": True,
            "id": 34,
            "supercategory": "vehicle"
        },
        {
            "color": [
                0,
                40,
                120
            ],
            "instances": True,
            "readable": "van (vehicle); ",
            "name": "van",
            "evaluate": True,
            "id": 35,
            "supercategory": "vehicle"
        },
        {
            "color": [
                174,
                64,
                67
            ],
            "instances": False,
            "readable": "billboard(object); sign boards, advertisements, etc. which are not traffic sign",
            "name": "billboard",
            "evaluate": True,
            "id": 36,
            "supercategory": "object"
        },
        {
            "color": [
                210,
                170,
                100
            ],
            "instances": False,
            "readable": "street-light(object); lamp plus housing part on top of poles or atttached to buildings",
            "name": "street-light",
            "evaluate": True,
            "id": 37,
            "supercategory": "object"
        },
        {
            "color": [
                196,
                176,
                128
            ],
            "instances": False,
            "readable": "road-marking (flat); includes all kinds of official markings on roads (including pedestrian crossing lines and hatched lines)",
            "name": "road-marking",
            "evaluate": True,
            "id": 38,
            "supercategory": "flat"
        }]
# wilddash -> city
wd_to_city_mappings = [
( 0 , 255 ),
( 1 , 255 ),
( 2 ,  255),
( 3 , 255 ),
( 4 , 255 ),
( 5 , 255 ),
( 6 , 255 ),
( 7 , 0 ),
( 8 , 1 ),
( 9 , 255 ),
( 10 , 255 ),
( 11 , 2 ),
( 12 , 3 ),
( 13 , 4 ),
( 14 , 255 ),
( 15 , 255 ),
( 16 , 255 ),
( 17 , 5 ),
( 18 , 255 ),
( 19 , 6 ),
( 20 , 7 ),
( 21 , 8 ),
( 22 , 9 ),
( 23 , 10 ),
( 24 , 11 ),
( 25 , 12 ),
( 26 , 13 ),
( 27 , 14 ),
( 28 , 15 ),
( 29 , 255 ),
( 30 , 255 ),
( 31 , 16 ),
( 32 , 17 ),
( 33 , 18 ),
( 34 , 13 ),
( 35 , 13 ),
( 36 , 255 ),
( 37 , 255 ),
( 38 , 0 )
]

def create_wilddash_to_cityscapes_mapper():
    mapper = torch.ones(len(wd_to_city_mappings)).long()
    for widl_class, city_class in wd_to_city_mappings:
        mapper[widl_class] = city_class
    mapper[mapper == 255] = 19
    return mapper


class ColorizeLabels:
    def __init__(self):
        color_info = [(label['id'], label['color']) for label in wilddash_config]
        self.color_info = dict()
        for (id, color) in color_info:
            self.color_info[id] = color

    def _trans(self, lab):
        R, G, B = [np.zeros_like(lab) for _ in range(3)]
        for l in np.unique(lab):
            mask = lab == l
            R[mask] = self.color_info[l][0]
            G[mask] = self.color_info[l][1]
            B[mask] = self.color_info[l][2]
        return torch.LongTensor(np.stack((R, G, B), axis=-1).astype(np.uint8)).squeeze().permute(2, 0, 1).float() / 255.

    def __call__(self, example):
        return self._trans(example)

colorize_labels = ColorizeLabels()

