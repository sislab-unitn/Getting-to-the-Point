import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import datasets
from PIL import Image
from torch.utils.data import Dataset
from world.world import World, ObjEntity


class MolmoDataset(Dataset):
    def __init__(
        self,
        results_file: str,
        predicted_samples: Set[str],
        ids_to_process: Optional[Set[str]] = None,
        ablate_image: bool = False,
        ablate_coords: bool = False,
        data_folder: str = "data/balanced",
    ):

        self.ablate_image = ablate_image
        self.ablate_coords = ablate_coords

        self.questions = []
        self.sample_ids = []
        self.answers = []
        self.images = []

        assert os.path.exists(results_file), f"{results_file} does not exist."

        with open(results_file, "r") as f:
            data = json.load(f)

        for img_id, entities in data.items():
            # Skip samples not in the subset of ids to process
            if ids_to_process and img_id not in ids_to_process:
                continue

            for ent_type, question_types in entities.items():
                for q_type, res in question_types.items():

                    if isinstance(res, dict):
                        for sub_q_type, sub_res in res.items():
                            sub_question = sub_res["input"]
                            answer = sub_res["coords"].split("shows a total of")[0]
                            answer += f"shows a total of {sub_res['target']}."

                            key = f"{img_id}-{ent_type}-{q_type}-{sub_q_type}"
                            if key not in predicted_samples:
                                self._add_sample(
                                    sample_id=key,
                                    question=sub_question,
                                    answer=answer,
                                    data_folder=data_folder,
                                    img_id=img_id,
                                )
                    else:
                        question = res["input"]
                        answer = res["coords"].split("shows a total of")[0]
                        answer += f"shows a total of {res['target']}."

                        key = f"{img_id}-{ent_type}-{q_type}"
                        if key not in predicted_samples:
                            self._add_sample(
                                sample_id=key,
                                question=question,
                                answer=answer,
                                data_folder=data_folder,
                                img_id=img_id,
                            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[Image.Image, str, str, str]:
        image = Image.open(self.images[idx])
        return (
            image,
            self.questions[idx],
            self.answers[idx],
            self.sample_ids[idx],
        )

    def _add_sample(
        self, sample_id: str, question: str, answer: str, data_folder: str, img_id: str
    ):
        if self.ablate_coords:  # TODO... change how this function works
            # remove coordinates from the answer
            coordinates, answer = answer.split(". Answer: ")
            coords = re.findall(r"\(\d+, \d+\)", coordinates)
            for i in range(len(coords)):
                ablated_coords = ", ".join(coords[:i] + coords[i + 1 :])
                self.sample_ids.append(f"{sample_id}-ablate{i}")
                self.questions.append(question)
                self.answers.append(f"Coordinates: {ablated_coords}. Answer: {answer}")
                self.images.append(os.path.join(data_folder, "images", f"{img_id}.png"))
        else:
            self.sample_ids.append(sample_id)
            self.questions.append(question)
            self.answers.append(answer)
            self.images.append(os.path.join(data_folder, "images", f"{img_id}.png"))

    def _make_open_ended(self, question: str) -> str:
        question = re.sub(r"Choose from \[.*\].", "", question).strip()
        assert (
            "[" not in question and "]" not in question
        ), f"squared brackets not removed from {question}"

        return question


class CIVETDataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        predicted_samples: Set[str],
        debug: bool = False,
        open_ended_questions: bool = False,
        ids_to_process: Optional[Set[str]] = None,
        add_coords: bool = False,
        add_distractors_coords: bool = False,
        add_xs: bool = False,
        ablate_image: bool = False,
        ablate_coords: bool = False,
    ):
        if ablate_coords:
            assert add_coords, "ablate_coords can only be used when add_coords is True"

        self.ablate_image = ablate_image
        self.ablate_coords = ablate_coords
        self.add_pointing_coords = add_coords

        if add_xs:
            assert not (
                add_coords or add_distractors_coords
            ), "add_xs cannot be used together with add_coords or add_distractors_coords"
        self.questions = []
        self.sample_ids = []
        self.answers = []
        self.images = []

        with open(os.path.join(data_folder, "questions.json"), "r") as f:
            data: Dict[str, Dict[str, Dict[str, Any]]] = json.load(f)

        with open(os.path.join(data_folder, "answers.json"), "r") as f:
            answers: Dict[str, Dict[str, Dict[str, Any]]] = json.load(f)

        if add_coords or add_distractors_coords or add_xs:
            assert os.path.exists(
                os.path.join(data_folder, "repr.json")
            ), "Adding coordinates requires the repr.json file."
            with open(os.path.join(data_folder, "repr.json"), "r") as f:
                repr = json.load(f)

        if debug:
            random_ids = random.sample(list(data.keys()), 100)

        for img_id, entities in data.items():
            # Skip samples not in the subset of ids when debugging
            if debug and img_id not in random_ids:
                continue

            # Skip samples not in the subset of ids to process
            if ids_to_process and img_id not in ids_to_process:
                continue

            for ent_type, question_types in entities.items():
                for q_type, question in question_types.items():

                    if add_coords or add_xs:
                        world = World.from_repr(repr[img_id])
                        ent_coords = []
                        for repr_ent_id in world.ents:
                            if repr_ent_id.startswith(("e", "t")):
                                ent: ObjEntity = world.ents[repr_ent_id]  # type: ignore
                                ent_coords.append(ent.position)

                        ent_coords = sorted(ent_coords)
                        coords_answer = ", ".join(
                            [f"({y}, {x})" for y, x in ent_coords]
                        )

                    if add_distractors_coords:
                        world = World.from_repr(repr[img_id])
                        distractor_coords = []
                        for repr_ent_id in world.ents:
                            if repr_ent_id.startswith("d"):
                                ent: ObjEntity = world.ents[repr_ent_id]  # type: ignore
                                distractor_coords.append(ent.position)

                        distractor_coords = sorted(distractor_coords)
                        distractors_coords_answer = ", ".join(
                            [f"({y}, {x})" for y, x in distractor_coords]
                        )

                    if isinstance(question, dict):
                        for sub_q_type, sub_question in question.items():
                            if open_ended_questions:
                                sub_question = self._make_open_ended(sub_question)
                            answer = ""
                            if add_distractors_coords:
                                answer += f"Distractor Coordinates: {distractors_coords_answer}. "
                            if add_coords:
                                answer += f"Coordinates: {coords_answer}. "
                            elif add_xs:
                                xs_answer = " ".join(["X" for _ in ent_coords])
                                answer += f"Coordinates: {xs_answer}. "

                            if add_coords or add_distractors_coords or add_xs:
                                answer += f"Answer: {answers[img_id][ent_type][q_type][sub_q_type]}"
                            else:
                                answer = answers[img_id][ent_type][q_type][sub_q_type]

                            key = f"{img_id}-{ent_type}-{q_type}-{sub_q_type}"
                            if key not in predicted_samples:
                                self._add_sample(
                                    sample_id=key,
                                    question=sub_question,
                                    answer=answer,
                                    data_folder=data_folder,
                                    img_id=img_id,
                                )
                    else:
                        if open_ended_questions:
                            question = self._make_open_ended(question)
                        answer = ""
                        if add_distractors_coords:
                            answer += (
                                f"Distractor Coordinates: {distractors_coords_answer}. "
                            )
                        if add_coords:
                            answer += f"Coordinates: {coords_answer}. "
                        elif add_xs:
                            xs_answer = " ".join(["X" for _ in ent_coords])
                            answer += f"Coordinates: {xs_answer}. "

                        if add_coords or add_distractors_coords or add_xs:
                            answer += f"Answer: {answers[img_id][ent_type][q_type]}"
                        else:
                            answer = answers[img_id][ent_type][q_type]

                        key = f"{img_id}-{ent_type}-{q_type}"
                        if key not in predicted_samples:
                            self._add_sample(
                                sample_id=key,
                                question=question,
                                answer=answer,
                                data_folder=data_folder,
                                img_id=img_id,
                            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[Image.Image, str, str, str]:
        image = Image.open(self.images[idx])
        if self.ablate_image:
            image = Image.new("RGB", image.size, (0, 0, 0))
        return (
            image,
            self.questions[idx],
            self.answers[idx],
            self.sample_ids[idx],
        )

    def _add_sample(
        self, sample_id: str, question: str, answer: str, data_folder: str, img_id: str
    ):
        if self.ablate_coords and self.add_pointing_coords:
            # remove coordinates from the answer
            coordinates, answer = answer.split(". Answer: ")
            coords = re.findall(r"\(\d+, \d+\)", coordinates)
            for i in range(len(coords)):
                ablated_coords = ", ".join(coords[:i] + coords[i + 1 :])
                self.sample_ids.append(f"{sample_id}-ablate{i}")
                self.questions.append(question)
                self.answers.append(f"Coordinates: {ablated_coords}. Answer: {answer}")
                self.images.append(os.path.join(data_folder, "images", f"{img_id}.png"))
        else:
            self.sample_ids.append(sample_id)
            self.questions.append(question)
            self.answers.append(answer)
            self.images.append(os.path.join(data_folder, "images", f"{img_id}.png"))

    def _make_open_ended(self, question: str) -> str:
        question = re.sub(r"Choose from \[.*\].", "", question).strip()
        assert (
            "[" not in question and "]" not in question
        ), f"squared brackets not removed from {question}"

        return question
