from collections import namedtuple
import torch
# --------------------------------------------------------------------------------
# Definitions
# --------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.

    'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!

    'eval',  # Whether pixels having this class as ground truth label are evaluated

    'color',  # The color of this label,
    'csTrainId'
])

# --------------------------------------------------------------------------------
# A list of all labels
# --------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                id       trainId    eval   color
    Label("unlabeled", 0, 255, 0, (0, 0, 0), 255),
    Label("ambiguous", 1, 255, 0, (111, 74, 0), 255),
    Label("sky", 2, 0, 1, (70, 130, 180), 10),
    Label("road", 3, 1, 1, (128, 64, 128), 0),
    Label("sidewalk", 4, 2, 1, (244, 35, 232), 1),
    Label("railtrack", 5, 255, 0, (230, 150, 140), 255),
    Label("terrain", 6, 3, 1, (152, 251, 152), 9),
    Label("tree", 7, 4, 1, (87, 182, 35), 8),
    Label("vegetation", 8, 5, 1, (35, 142, 35), 8),
    Label("building", 9, 6, 1, (70, 70, 70), 2),
    Label("infrastructure", 10, 7, 1, (153, 153, 153), 255),
    Label("fence", 11, 8, 1, (190, 153, 153), 4),
    Label("billboard", 12, 9, 1, (150, 20, 20), 255),
    Label("trafficlight", 13, 10, 1, (250, 170, 30), 6),
    Label("trafficsign", 14, 11, 1, (220, 220, 0), 7),
    Label("mobilebarrier", 15, 12, 1, (180, 180, 100), 255),
    Label("firehydrant", 16, 13, 1, (173, 153, 153), 255),
    Label("chair", 17, 14, 1, (168, 153, 153), 255),
    Label("trash", 18, 15, 1, (81, 0, 21), 255),
    Label("trashcan", 19, 16, 1, (81, 0, 81), 255),
    Label("person", 20, 17, 1, (220, 20, 60), 255),
    Label("animal", 21, 255, 0, (255, 0, 0), 255),
    Label("bicycle", 22, 255, 0, (119, 11, 32), 18),
    Label("motorcycle", 23, 18, 1, (0, 0, 230), 17),
    Label("car", 24, 19, 1, (0, 0, 142), 13),
    Label("van", 25, 20, 1, (0, 80, 100), 255),
    Label("bus", 26, 21, 1, (0, 60, 100), 15),
    Label("truck", 27, 22, 1, (0, 0, 70), 14),
    Label("trailer", 28, 255, 0, (0, 0, 90), 255),
    Label("train", 29, 255, 0, (0, 80, 100), 16),
    Label("plane", 30, 255, 0, (0, 100, 100), 255),
    Label("boat", 31, 255, 0, (50, 0, 90), 255)
]


def get_train_ids():
    train_ids = []
    for i in labels:
        if i.eval:
            train_ids.append(i.id)
    return train_ids


def get_class_info():
    class_info = []
    for i in labels:
        if i.eval:
            class_info.append([i.color[0], i.color[1], i.color[2], i.name])
    return class_info

def create_id_to_train_id_mapper():
    lookup_table = torch.ones(256).long() * 255
    for label in labels[:-1]:
        lookup_table[label[1]] = label[-1]
    lookup_table[lookup_table == 255] = 255
    return lookup_table