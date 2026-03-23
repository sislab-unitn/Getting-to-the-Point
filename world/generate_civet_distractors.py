from collections import Counter
import json
import os
import argparse
import random

from statistics import mean, stdev
from itertools import chain, permutations, product
from typing import Any, Callable, Dict, Iterable, List, Set, Tuple
from PIL import Image
from time import time_ns
from tqdm import tqdm
from questions import PROP_VALUES, PROPERTIES_ORDER, absolute_position_question, prop_question, properties_questions
from sprites import SpritesLoader
from utils import COLORS, SHEENS, SHAPES, SIZES, get_uniform_samples, random_value_order
from world import World


def set_diff(values: Iterable[Any], v: Any) -> Set[Any]:
    return set(values).difference([v])

def pos_area_diff(positions: List[Tuple[int, int]], pos: Tuple[int, int], w: World, **kwargs) -> Set[Tuple[int, int]]:
    # w = World((size, size))

    pos_area = w.area_map.get_area(pos)
    area_positions = set(w.area_map.get_area_positions(pos_area))
    assert pos in area_positions

    diff_positions = set(positions).difference(area_positions)
    assert w.size[0]*w.size[1] - len(area_positions) == len(diff_positions)

    return diff_positions

def get_distr_samples_by_prop(
    values: list,
    n_distractors: int,
    n_variations: int,
    n_samples: int,
    get_diff: Callable[[Iterable[Any], Any], Iterable[Any]] = set_diff,
    **kwargs
) -> Dict[str, List[Any]]:
    samples = {}
    for v in tqdm(values, desc="Generating distractors", unit="values"):
        # diff_values = set(values).difference([v])
        diff_values = get_diff(values, v, **kwargs)
        assert v not in diff_values
        values_combs = list(product(diff_values, repeat=n_distractors))
        # for each target object, need n_variations samples
        samples[v] = get_uniform_samples(values_combs, n=n_samples*n_variations)
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument("--img_size", type=int, default=336)
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--n_distractors", type=int, default=3, help="Number of distractors for each image")
    parser.add_argument("--n_variations", type=int, default=3, help="Number of images with distractors to generate from the original image containing only the target object")
    args = parser.parse_args()

    os.makedirs(args.name, exist_ok=True)

    world_size = [args.size, args.size]

    sprites_loader = SpritesLoader()

    worlds_repr = {}
    questions = {}

    positions = [(i,j) for i in range(args.size) for j in range(args.size)]

    # property variations for one object
    shapes = ["square", "circle", "triangle", "star"]
    colors = ["red", "green", "blue", "magenta", "yellow", "cyan"]
    # all shapes, white
    prop_vairations =  list(product(shapes, ["white"], SIZES, positions))
    # plus, all colors
    prop_vairations += list(product(["plus"], colors, SIZES, positions))

    # duplicate target for each distractor variation
    prop_vairations = list(chain(*[[variation]*args.n_variations for variation in prop_vairations]))

    n_stimuli = len(prop_vairations)


    # Distractors
    random.seed(42)
    test_w = World((args.size, args.size))

    # sizes
    sizes_combs = list(product(SIZES, repeat=args.n_distractors))
    sizes_distr = get_uniform_samples(sizes_combs, n=n_stimuli)

    # white shapes
    shapes_distr = get_distr_samples_by_prop(shapes, args.n_distractors, args.n_variations, n_samples=(args.size**2)*len(SIZES))
    ws_positions_distr = get_distr_samples_by_prop(positions, args.n_distractors, args.n_variations, n_samples=len(shapes)*len(SIZES), get_diff=pos_area_diff, w=test_w)

    # colored plusses
    colors_distr = get_distr_samples_by_prop(colors, args.n_distractors, args.n_variations, n_samples=(args.size**2)*len(SIZES))
    cp_positions_distr = get_distr_samples_by_prop(positions, args.n_distractors, args.n_variations, n_samples=len(colors)*len(SIZES), get_diff=pos_area_diff, w=test_w)


    print((args.size**2)*len(SIZES)*len(shapes), len(shapes)*len(SIZES)*len(positions), len(shapes)*(args.size**2)*len(SIZES), (args.size**2)*len(SIZES)*len(colors), len(colors)*len(SIZES)*len(positions), len(colors)*(args.size**2)*len(SIZES))
    print((args.size**2)*len(SIZES)*len(shapes) * args.n_variations)



    n_shapes = len(PROP_VALUES["shape"])
    n_colors = len(PROP_VALUES["color"])
    shape_orders = random_value_order(n_stimuli, n_shapes)
    color_orders = random_value_order(n_stimuli, n_colors)

    n_areas = len(PROP_VALUES["area"])
    n_areas_vert = len(PROP_VALUES["area_vert"])
    n_area_hors = len(PROP_VALUES["area_hor"])
    area_orders = random_value_order(n_stimuli, n_areas)
    area_vert_orders = random_value_order(n_stimuli, n_areas_vert)
    area_hor_orders = random_value_order(n_stimuli, n_area_hors)

    for name, orders in zip(
        ["shape", "color", "area", "area_vert", "area_hor"],
        [shape_orders, color_orders, area_orders, area_vert_orders, area_hor_orders]
    ):
        print("-"*40)
        print(name)
        c = Counter(orders)
        print(f"{mean(c.values())} +- {stdev(c.values())}, max-min: {max(c.values()) - min(c.values())}")


    variations = []
    i = 0
    for v, shape_ord, color_ord, area_ord, area_vert_ord, area_hor_ord in zip(prop_vairations, shape_orders, color_orders, area_orders, area_vert_orders, area_hor_orders):
        variation = [v, shape_ord, color_ord]
        variation.extend([area_ord, area_vert_ord, area_hor_ord])
        variations.append(tuple(chain(variation)))

    assert len(variations) == n_stimuli

    for stim_n, (variation, shape_ord, color_ord, area_ord, area_vert_ord, area_hor_ord) in tqdm(enumerate(variations), desc=f"Generating stimuli", unit="stimuli", total=len(variations)):

        params = {
            "size": world_size,
            "img_size": args.img_size,
            "background": "none",
            "objects": [],
        }

        # for shape, color, sheen, size, position in variation:
        shape, color, size, position = variation
        params["objects"].append({
            "count": 1,
            "shape": shape,
            "color": color,
            "sheen": "none",
            "size": size,
            "position": position
        })

        # add distractors
        if color == "white":
            # for white shapes
            distr_shapes = shapes_distr[shape].pop(0)
            distr_colors = tuple(["white"]*3)
            distr_positions = ws_positions_distr[position].pop(0)
            # distr_sizes = ws_size_distr[size].pop(0)
        else:
            # for colored plusses
            distr_shapes = tuple(["plus"]*3)
            distr_colors = colors_distr[color].pop(0)
            distr_positions = cp_positions_distr[position].pop(0)
            # distr_sizes = cp_size_distr[size].pop(0)
        distr_sizes = sizes_distr.pop(0)

        for d_shape, d_color, d_size, d_position in zip(distr_shapes, distr_colors, distr_sizes, distr_positions):
            assert (
                (d_shape != shape or shape == "plus") and
                (d_color != color or color == "white") and
                test_w.area_map.get_area(d_position) != test_w.area_map.get_area(position)
            )
            params["objects"].append({
                "count": 1,
                "shape": d_shape,
                "color": d_color,
                "sheen": "none",
                "size": d_size,
                "position": d_position
            })

        grid_size = tuple(params["size"])

        background = "none"
        if "background" in params:
            background = params["background"]

        objects = params["objects"]

        world = World(grid_size, background=background)

        for obj_template in objects:
            world.add(**obj_template)

        s = world.get_stimulus(sprites_loader, img_size=args.img_size)
        save_folder = os.path.join(args.name, "images")
        save_path = os.path.join(save_folder, f"{stim_n}.png")
        os.makedirs(save_folder, exist_ok=True)
        Image.fromarray(s).save(save_path)

        worlds_repr[stim_n] = world.to_dict(img_path=save_path)
        # question about the first entity
        e = world.ents["e1"]
        if stim_n not in questions:
            questions[stim_n] = {}
        if e.id not in questions[stim_n]:
            questions[stim_n][e.id] = {}
        questions[stim_n][e.id]["position_absolute"] = absolute_position_question(e, val_order=area_ord)
        questions[stim_n][e.id]["position_absolute_vert"] = absolute_position_question(e, val_order=area_vert_ord, val_list_name="area_vert")
        questions[stim_n][e.id]["position_absolute_hor"] = absolute_position_question(e, val_order=area_hor_ord, val_list_name="area_hor")

    with open(os.path.join(args.name, "repr.json"), "w") as f:
        json.dump(worlds_repr, f, indent=4)
    with open(os.path.join(args.name, "questions.json"), "w") as f:
        json.dump(questions, f, indent=4)

if __name__ == "__main__":
    main()
