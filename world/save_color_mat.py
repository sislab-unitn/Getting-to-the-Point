import os
import numpy as np
from typing import List, Optional, Tuple
from PIL import Image
from numpy._typing import NDArray
from utils import COLOR_MAP, COLORS, MATERIALS, SHAPES, change_color, load_image, resize

def substitute_rgba_values(sprite: NDArray, old_color: Tuple[int, int, int, int], new_color: Tuple[int, int, int, int], thresh: int = 20):
    r,g,b,a = sprite.T
    # old_r, old_g, old_b, old_a = old_color
    # mask = ((r >= old_r - thresh) & (r <= old_r + thresh)) & (g == old_g) & (b == old_b) & (a == old_a)
    mask = np.full(r.shape,True)
    for ch, old_c in zip([r,g,b,a], old_color):
        mask = mask & ((ch >= max(old_c - thresh, 0)) & (ch <= min(old_c + thresh, 255)))
    sprite[mask.T] = new_color


def apply_material(img: NDArray, material: NDArray) -> NDArray:
    r,g,b,a = img.T
    mask = g == 255
    mat_img = np.full(img.shape, [0,0,0,0]).astype(np.uint8)
    mat_img[mask.T] = material[mask.T]
    return mat_img


def load_sprite(path: str, size: Optional[Tuple[int, int]] = None, replace_colors: Optional[List[Tuple[Tuple[int, int, int ,int], Tuple[int, int, int, int]]]] = None, thresh: int = 20, material_path: Optional[str] = None, border_ratio: float = 0.14) -> NDArray:
    img = load_image(path, border_ratio=border_ratio)

    # apply material
    if material_path is not None:
        material = load_image(material_path)
        img = Image.fromarray(apply_material(img, material))


    if replace_colors is not None:
        img = np.array(img)
        for old_color, new_color in replace_colors:
            # TODO adapt for textures (might just have to combine texture and color)
            # substitute_rgba_values(img, old_color=old_color, new_color=new_color, thresh=thresh)
            # change_color(img, old_color=old_color, new_color=new_color, thresh=thresh)
            img = change_color(img, old_color, new_color)
        img = Image.fromarray(img)
    if size is not None:
        img = resize(img, size)
    else:
        img = np.array(img)
    return img


def main():
    sprites = {}

    # load squares with material color changed and no border
    sprites["objects"] = {
        shape: {
            color: {
                material: load_sprite(
                    os.path.join("./assets", "shapes_new", f"{shape}.svg"),
                    replace_colors=[(tuple([int(round(c/255)) for c in COLOR_MAP["placeholder"]]), tuple([int(round(c/255)) for c in COLOR_MAP[color]]))],
                    material_path=os.path.join("./assets", "blender", f"{material}.png"),
                    border_ratio=0.0
                )
                for material in MATERIALS
            }
            for color in COLORS
        }
        for shape in SHAPES
    }

    for color, materials in sprites["objects"]["square"].items():
        for material, img in materials.items():
            save_folder = os.path.join("./assets", "materials")
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, f"{color}_{material}.png")
            Image.fromarray(img).save(save_path)


if __name__ == "__main__":
    main()
