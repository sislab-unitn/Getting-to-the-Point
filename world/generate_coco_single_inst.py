from collections import Counter
import json
import os
import argparse
import random
import math
import shutil

from statistics import mean, stdev
from itertools import chain, permutations, product
from typing import Any, Dict, List, Tuple
from PIL import Image
from time import time_ns
from tqdm import tqdm
from questions import PROP_VALUES, PROPERTIES_ORDER, absolute_position_question, prop_question, properties_questions
from sprites import SpritesLoader
from utils import COLORS, SHEENS, SHAPES, SIZES, CATEGORIES, random_value_order
from world import World

from pycocotools.coco import COCO
from collections import Counter



def load_coco_annotation(splits: List[str] = ["val"], annotation_folder: str = "../coco/annotations") -> Dict[str, Any]:
    annotation = {}
    for split in splits:
        annotation_file = os.path.join(annotation_folder, f"instances_{split}2017.json")
        annotation[split] = COCO(annotation_file=annotation_file)
    return annotation

def get_pos(img, obj_ann, grid_size: int) -> Tuple[float, float]:
    x, y, w, h = map(int, obj_ann['bbox'])
    c_x, c_y = (x+(w/2), y+(h/2))
    x_area, y_area = (math.floor(grid_size*c_x/img.size[0]), math.floor(grid_size*c_y/img.size[1]))
    return x_area, y_area

def get_coco_data(annotation: Dict[str, Any], data_folder: str = "../coco/", grid_size: int = 9):
    single_cat_instance_counts = {}
    single_cat_instance_examples = {}

    for split in annotation.keys():
        n_images = len(annotation[split].getImgIds())

        n_annotations = len(annotation[split].getAnnIds())

        # count images having only one object of a given category
        single_cat_instance_counts[split] = Counter()
        single_cat_instance_img_counts = 0
        single_cat_instance_examples[split] = []
        for img in tqdm(annotation[split].getImgIds(), desc="Extracting objects by cat in images", unit="images"):
            ann_ids = annotation[split].getAnnIds(imgIds=img)
            anns = annotation[split].loadAnns(ann_ids)
            cats = annotation[split].loadCats([ann['category_id'] for ann in anns])

            img_file = annotation[split].loadImgs(img).pop()['file_name']
            img_path = os.path.join(data_folder, f"{split}2017/{img_file}")
            img = Image.open(img_path)

            objects = {}
            for ann, cat in zip(anns, cats):
                category = cat["name"]
                if category not in objects:
                    objects[category] = []

                # position
                x_area, y_area = get_pos(img, ann, grid_size)
                # row, column
                objects[category].append((y_area, x_area))

            img_cat_counts = {
                cat:len(positions)
                for cat, positions in objects.items()
                if len(positions) == 1
            }
            single_cat_instance_counts[split].update(img_cat_counts)

            # count images that have at least one category
            # with only one instance in the image
            if len(img_cat_counts) > 0:
                single_cat_instance_img_counts += 1
                single_cat_instance_examples[split].append({
                    "path": img_path,
                    "objects": objects
                })

        single_cat_instance_img_counts_perc = 100*single_cat_instance_img_counts/n_images
        # table[split]["# Images w/ single instances of categories"] = f"{single_cat_instance_img_counts} ({round(single_cat_instance_img_counts_perc)}%)"
        single_cat_inst = sum(single_cat_instance_counts[split].values())
        single_cat_inst_perc = 100*single_cat_inst/n_annotations
        # table[split]["# Obj single instances of categories"] = f"{single_cat_inst} ({round(single_cat_inst_perc)}%)"

        print(f"# images with single instance of category (% of split): {single_cat_instance_img_counts} ({single_cat_instance_img_counts_perc:.2f}%)")
        print(f"# images with single instance by cat: {single_cat_instance_counts[split]}")
        print(f"# single category instances (% of split): {single_cat_inst} ({single_cat_inst_perc:.2f}%)")
    return single_cat_instance_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str)
    # parser.add_argument("--img_size", type=int, default=336)
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--test", action="store_true", default=9)
    args = parser.parse_args()


    # load coco data
    annotation = load_coco_annotation()

    coco_data = get_coco_data(annotation, grid_size=args.size)
    for split, images in coco_data.items():
        for img_dict in images:
            assert 1 in [len(coords) for cat, coords in img_dict["objects"].items()]

    assert len(coco_data.keys()) and "val" in coco_data
    coco_data = coco_data["val"]


    os.makedirs(args.name, exist_ok=True)

    world_size = [args.size, args.size]


    worlds_repr = {}
    questions = {}

    n_stimuli = sum([sum([len(positions) for cat, positions in img_dict["objects"].items() if len(positions) == 1]) for img_dict in coco_data])
    print("# single instances / n_stimuli: ", n_stimuli)

    n_areas = len(PROP_VALUES["area"])
    n_areas_vert = len(PROP_VALUES["area_vert"])
    n_area_hors = len(PROP_VALUES["area_hor"])
    area_orders = random_value_order(n_stimuli, n_areas)
    area_vert_orders = random_value_order(n_stimuli, n_areas_vert)
    area_hor_orders = random_value_order(n_stimuli, n_area_hors)

    for name, orders in zip(
        ["area", "area_vert", "area_hor"],
        [area_orders, area_vert_orders, area_hor_orders]
    ):
        print("-"*40)
        print(name)
        c = Counter(orders)
        print(f"{mean(c.values())} +- {stdev(c.values())}, max-min: {max(c.values()) - min(c.values())}")


    variations = []
    i = 0
    for area_ord, area_vert_ord, area_hor_ord in zip(area_orders, area_vert_orders, area_hor_orders):
        # variation = [category_ord]
        variation = []
        variation.extend([area_ord, area_vert_ord, area_hor_ord])
        variations.append(tuple(chain(variation)))

    assert len(variations) == n_stimuli

    ref_properties = ["category"]

    stim_n = 0
    test_c = 0
    for world_n, img_dict in enumerate(coco_data):
        test_c +=1
        if args.test and test_c > 100:
            break
        # single categories for image
        single_categories = [cat for cat, positions in img_dict["objects"].items() if len(positions) == 1]

        area_ord, area_vert_ord, area_hor_ord = variations[stim_n]

        params = {
            "size": world_size,
            # "img_size": "TODO", #args.img_size, # TODO resize or use img size
            "background": "none",
            "objects": [],
        }

        # add all objects
        for cat, positions in img_dict["objects"].items():
            # if other_cat == cat:
            #     continue
            for pos in positions:
                params["objects"].append({
                    "count": 1,
                    "category": cat,
                    # "size": "TODO", # TODO
                    "position": pos
                })

        grid_size = tuple(params["size"])

        background = "none"
        if "background" in params:
            background = params["background"]


        objects = params["objects"]

        world = World(grid_size, background=background)

        for obj_template in objects:
            world.add(**obj_template)

        save_folder = os.path.join(args.name, "images")
        save_path = os.path.join(save_folder, f"{world_n}.png")
        os.makedirs(save_folder, exist_ok=True)
        shutil.copyfile(img_dict["path"], save_path)

        worlds_repr[world_n] = world.to_dict(img_path=save_path)
        for e_id, e in world.ents.items(): # TODO class for COCO obj
            if e_id == "w":
                continue
            if e.category not in single_categories:
                continue

            if world_n not in questions:
                questions[world_n] = {}
            if e.id not in questions[world_n]:
                questions[world_n][e.id] = {}
            questions[world_n][e.id]["position_absolute"] = absolute_position_question(e, val_order=area_ord, properties=ref_properties)
            questions[world_n][e.id]["position_absolute_vert"] = absolute_position_question(e, val_order=area_vert_ord, val_list_name="area_vert", properties=ref_properties)
            questions[world_n][e.id]["position_absolute_hor"] = absolute_position_question(e, val_order=area_hor_ord, val_list_name="area_hor", properties=ref_properties)

            stim_n += 1

    with open(os.path.join(args.name, "repr.json"), "w") as f:
        json.dump(worlds_repr, f, indent=4)
    with open(os.path.join(args.name, "questions.json"), "w") as f:
        json.dump(questions, f, indent=4)


if __name__ == "__main__":
    main()
