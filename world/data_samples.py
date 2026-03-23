import argparse
import os
import json
import random
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_folder", type=str)
    parser.add_argument("--n_samples", type=int, default=10)
    args = parser.parse_args()

    img_folder = os.path.join(args.data_folder, "images")
    questions_file = os.path.join(args.data_folder, "questions.json")
    repr_file = os.path.join(args.data_folder, "repr.json")

    with open(questions_file, "r") as f:
        all_questions = json.load(f)

    with open(repr_file, "r") as f:
        all_repr = json.load(f)

    ids = all_repr.keys()
    sampled_ids = random.sample(list(ids), k=args.n_samples)
    sampled_ids = sorted([int(id) for id in sampled_ids])
    sampled_ids = [f"{id}" for id in sampled_ids]

    save_folder = os.path.join("samples", args.data_folder)
    os.makedirs(save_folder)

    save_img_folder = os.path.join(save_folder, "images")
    os.makedirs(save_img_folder, exist_ok=True)

    questions = {}
    repr = {}
    for id in sampled_ids:
        questions[id] = all_questions[id]
        repr[id] = all_repr[id]
        shutil.copyfile(os.path.join(img_folder, f"{int(id)}.png"), os.path.join(save_img_folder, f"{int(id)}.png"))

    with open(os.path.join(save_folder, "questions.json"), "w") as f:
        json.dump(questions, f, indent=4)
    with open(os.path.join(save_folder, "repr.json"), "w") as f:
        json.dump(repr, f, indent=4)



if __name__ == "__main__":
    main()
