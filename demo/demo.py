# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/demo/demo.py
# Modified by Jitesh Jain (https://github.com/praeclarumjj3)
# ------------------------------------------------------------------------------

import argparse
import multiprocessing as mp
import os
import torch
import random
# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import time
import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)
from predictor import VisualizationDemo


# importing for endpoint
import json
import numpy
import pycocotools.mask  as _mask
from imantics import Polygons, Mask
from detectron2.structures import Boxes

# constants
WINDOW_NAME = "OneFormer Demo"

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_dinat_config(cfg)
    add_convnext_config(cfg)
    add_oneformer_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="oneformer demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="../configs/ade20k/swin/oneformer_swin_large_IN21k_384_bs16_160k.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--task", help="Task type")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def oneformer_output():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    info = None

    if args.input:
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
                
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img, args.task)
            info = predictions["instances"]
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )
            if args.output:
                if len(args.input) == 1:
                    for k in visualized_output.keys():
                        os.makedirs(k, exist_ok=True)
                        out_filename = os.path.join(k, args.output)
                        visualized_output[k].save(out_filename)    
                else:
                    for k in visualized_output.keys():
                        opath = os.path.join(args.output, k)    
                        os.makedirs(opath, exist_ok=True)
                        out_filename = os.path.join(opath, os.path.basename(path))
                        visualized_output[k].save(out_filename)    
            else:
                raise ValueError("Please specify an output path!")
    else:
        raise ValueError("No Input Given")
    return info


''''''''''
Descprition: this is where we will be running the code that will 
call oneformer and then where we will parse the information 
into a json and then save it into a json
'''
if __name__ == "__main__":

    # these are the classes for coco 
    idToObject = {0 : "person",
              1 : "Bicycle",
              2 : "car",
              3 : "Motorcycle",
              4 : "Airplace",
              5 : "Bus",
              6 : "Train",
              7 : "Truck",
              8 : "Boat",
              9 : "Traffic Light",
              10 : "Fire Hydrant",
              11 : "Stop sign",
              12 : "Parking Meter",
              13 : "Bench",
              14 : "Bird",
              15 : "Cat",
              16 : "Dog",
              17 : "Horse",
              18 : "Sheep",
              19 : "Cow",
              20 : "Elephant",
              21 : "Bear",
              22 : "Zebra",
              23 : "Giraffe",
              24 : "Backpack",
              25 : "Umbrella",
              26 : "Handbag",
              27 : "Tie",
              28 : "Suitcase",
              29 : "Frisbee",
              30 : "Skis",
              31 : "Snowboard",
              32 : "Sports ball",
              33 : "Kite",
              34 : "Baseball bat",
              35 : "Baseball glove",
              36 : "Skateboard",
              37 : "Surfboard",
              38 : "Tennis racket",
              39 : "Bottle",
              40 : "Wine glass",
              41 : "Cup",
              42 : "Fork",
              43 : "Knife",
              44 : "Spoon",
              45 : "Bowl",
              46 : "Banana",
              47 : "Apple",
              48 : "Sandwich",
              49 : "Orange",
              50 : "Broccoli",
              51 : "Carrot",
              52 : "Hot dog",
              53 : "Pizza",
              54 : "Donut",
              55 : "Cake",
              56 : "Chair",
              57 : "Couch",
              58 : "Potted plant",
              59 : "Bed",
              60 : "Dining table",
              61 : "Toilet",
              62 : "TV",
              63 : "Laptop",
              64 : "Mouse",
              65 : "Remote",
              66 : "Keyboard",
              67 : "Cell phone",
              68 : "Microwave",
              69 : "Oven",
              70 : "Toaster",
              71 : "Sink",
              72 : "Refrigerator",
              73 : "Book",
              74 : "Clock",
              75 : "Vase",
              76 : "Scissors",
              77 : "Teddy bear",
              78 : "Hair drier",
              79 : "Toothbrush",
            }


    # this is where we will get the output from oneformer 
    results = oneformer_output()

    # this is the information that we will need to recieve from oneformer 
    pred_classes = results.pred_classes.cpu().numpy()
    scores = results.scores.cpu().numpy()
    pred_mask = results.pred_masks.cpu().numpy()
    pred_boxes = results.pred_boxes.tensor.numpy()

    # this is where we will be storing the data
    data = {}

    # this is used for the cases where we might have multiple objects with the same name 
    name_count = {}

    # here we will be stroing all of the objects that we have found with their names
    index = 0
    for name in pred_classes:
        obj_name = idToObject[name]

        if obj_name in name_count:
            name_count[obj_name] = name_count[obj_name] + 1
            data[index] = {"name" : obj_name + str(name_count[obj_name])}
        else:
            name_count[obj_name] = 1
            data[index] = {"name" : obj_name + str(name_count[obj_name])}
        index = index + 1

    print("Data: ", data)

    # now we are going to put in score information 
    index = 0
    for score in scores:
        data[index]["score"] = float(score)
        index = index + 1
    
    # now we are going to add in the pred_box
    index = 0
    for bbox in pred_boxes:
        coord_list = bbox.tolist()

        x_min = coord_list[0]
        y_min = coord_list[1]
        x_max = coord_list[2]
        y_max = coord_list[3]

        # here we will be getting the bounindg box here 
        data[index]["bbox"] = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]

        index = index + 1

    # now we are going to add in the mask
    index = 0
    for mask in pred_mask:
        binary_mask = mask
        contours, _ = cv2.findContours((binary_mask * 255).astype(numpy.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            verticies = largest_contour.reshape(-1, 2).tolist()

        data[index]["mask"] = verticies

        index = index + 1 

    # this is the json that we will later store out information into 
    json_results = {"results" : []}

    # put in all of the objects that we have so far 
    for object in data:
        json_results["results"].append(data[object])

    # store the json into a json that we will later read from 
    json_data = json.dumps(json_results, indent=4)
    with open("output.json", "w") as json_file:
        json_file.write(json_data)

