import argparse
import json
import os
import random
from functools import partial

import torch
import numpy as np
from peft.peft_model import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor


from utils.data import CIVETDataset
from utils.metrics import (
    aggregate_results_per_ent_type,
    create_classification_report,
    create_confusion_matrix,
    match_number_in_text,
)
from utils.utils import MAP_MODELS, generate


def configure_subparsers(subparsers: argparse._SubParsersAction):
    """Configure a new subparser ."""
    parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate the models performance on CIVET.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "data_folder",
        metavar="DATA_FOLDER",
        type=str,
        help="Folder containing train, validation, and test splits.",
    )
    parser.add_argument(
        "experiment_name",
        metavar="EXPERIMENT_NAME",
        type=str,
        help="Name of the experiment.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use only 100 questions to check the answer of the model.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1000,
        help="Save the results after every N steps.",
    )
    parser.add_argument(
        "--instruction",
        metavar="INSTR",
        type=str,
        default="Answer using as few words as possible",
        help="Instruction used for generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        metavar="N_TOKENS",
        type=int,
        default=5,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--open-ended-questions",
        "-oq",
        action="store_true",
        help="Use open ended questions.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Override the existing results.",
    )
    parser.add_argument(
        "--model-checkpoint",
        metavar="MODEL_CHECKPOINT",
        type=str,
        default=None,
        help="Path to a model checkpoint to load.",
    )
    parser.add_argument(
        "--ids-to-process",
        metavar="VALID_IDS",
        nargs="+",
        type=str,
        default=None,
        help="List of valid ids to considered.",
    )

    parser.set_defaults(func=main)


def main(args):
    # get the model and processor based on the model name
    model_name, Model, Collator = MAP_MODELS[args.model_name]

    model = Model(
        model_name,
        low_cpu_mem_usage=True,
        device_map="auto" if args.parallel else args.device,
    )
    Processor = partial(AutoProcessor.from_pretrained, model_name, use_fast=True)

    if (
        "llava" in args.model_name
        or "qwen" in args.model_name
        or "internvl3" in args.model_name
        or args.model_name in ["paligemma2-10b", "molmo-o-7b"]
    ):
        if "qwen2" in args.model_name:
            processor = Processor(
                min_pixels=100 * 28 * 28,
                max_pixels=2304 * 28 * 28,  # size of 1344x1344 images
            )
        elif args.model_name in ["molmo-o-7b", "llava-onevision-8b"]:
            processor = Processor(trust_remote_code=True)
        else:
            processor = Processor()

        civet_collator = Collator(processor, args.instruction)
        if args.model_name == "paligemma2-10b":
            torch.set_float32_matmul_precision("high")
    else:
        raise ValueError(f"Invalid model name: {args.model_name}")

    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"

    if args.model_checkpoint is not None:
        print(f"Loading model from checkpoint: {args.model_checkpoint}")
        model = PeftModel.from_pretrained(model, args.model_checkpoint)

    # model config
    model.eval()

    # set the seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    predicted_samples = []
    if (
        os.path.exists(os.path.join(args.out_dir, f"predicted_samples.json"))
        and not args.override
    ):
        with open(os.path.join(args.out_dir, f"predicted_samples.json"), "r") as f:
            predicted_samples = json.load(f)

    test_ds = CIVETDataset(
        data_folder=args.data_folder,
        predicted_samples=set(predicted_samples),
        debug=args.debug,
        open_ended_questions=args.open_ended_questions,
        ids_to_process=args.ids_to_process,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=(1 if args.model_name == "molmo-o-7b" else args.batch_size),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=civet_collator,
    )

    results = {}
    if os.path.exists(os.path.join(args.out_dir, f"results.json")):
        with open(os.path.join(args.out_dir, f"results.json"), "r") as f:
            results = json.load(f)

    current_step = len(predicted_samples)
    with torch.no_grad():
        for input_ids, input_texts, targets, sample_ids in tqdm(
            test_loader, desc=f"Counting Objects"
        ):
            preds = generate(
                input_ids, model, args.model_name, processor, args.max_new_tokens  # type: ignore
            )

            for i, (sample_id, pred, target) in enumerate(
                zip(sample_ids, preds, targets)
            ):

                if args.model_name == "molmo-o-7b":
                    input = processor.tokenizer.decode(  # type: ignore
                        input_texts["input_ids"][i], skip_special_tokens=True
                    )
                else:
                    input = processor.decode(input_texts.input_ids[i], skip_special_tokens=True)  # type: ignore
                original_pred = pred
                pred = match_number_in_text(pred)  # type: ignore

                res = {
                    "target": target,
                    "input": input,
                    "original_pred": original_pred,
                    "pred": pred,
                }

                img_id, ent_type, *q_type = sample_id.split("-")

                if img_id not in results:
                    results[img_id] = {}

                if ent_type not in results[img_id]:
                    results[img_id][ent_type] = {}

                if len(q_type) == 2:
                    q_type, sub_q_type = q_type
                    if q_type not in results[img_id][ent_type]:
                        results[img_id][ent_type][q_type] = {}

                    results[img_id][ent_type][q_type][sub_q_type] = res
                elif len(q_type) == 1:
                    # unpack it
                    q_type = q_type[0]
                    results[img_id][ent_type][q_type] = res
                else:
                    raise ValueError(f"Invalid question type: {q_type}")

                current_step += 1
                if current_step % args.save_every == 0:
                    with open(os.path.join(args.out_dir, f"results.json"), "w") as f:
                        json.dump(results, f, indent=4)

                    with open(
                        os.path.join(args.out_dir, f"predicted_samples.json"), "w"
                    ) as f:
                        json.dump(predicted_samples, f, indent=4)

                # add the predicted samples to keep track of them
                predicted_samples.append(sample_id)

    with open(os.path.join(args.out_dir, f"results.json"), "w") as f:
        json.dump(results, f, indent=4)

    with open(os.path.join(args.out_dir, f"predicted_samples.json"), "w") as f:
        json.dump(predicted_samples, f, indent=4)

    y_true = []
    y_pred = []
    summary = {}
    results_per_ent_type = aggregate_results_per_ent_type(results)
    for ent_type in results_per_ent_type:
        ent_y_true = results_per_ent_type[ent_type]["y_true"]
        ent_y_pred = results_per_ent_type[ent_type]["y_pred"]

        report = create_classification_report(
            ent_y_true, ent_y_pred, f"{args.out_dir}/{ent_type}"
        )

        create_confusion_matrix(ent_y_true, ent_y_pred, f"{args.out_dir}/{ent_type}")

        y_true.extend(ent_y_true)
        y_pred.extend(ent_y_pred)
        summary[ent_type] = {"accuracy": report["accuracy"]}

    create_classification_report(y_true, y_pred, args.out_dir)

    create_confusion_matrix(y_true, y_pred, args.out_dir)

    with open(os.path.join(args.out_dir, f"summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
