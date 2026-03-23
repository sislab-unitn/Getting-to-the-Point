import argparse
from functools import partial
import gc
import json
import os
import random
from tqdm import tqdm


import torch
import numpy as np
from peft import LoraConfig, get_peft_model  # type: ignore
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from transformers import AutoProcessor


from utils.data import CIVETDataset
from utils.utils import MAP_MODELS_TRAIN, seed_worker
from utils.training import (
    Checkpointer,
    load_training_params,
    resume_training,
    save_training_params,
    train,
)


def configure_subparsers(subparsers: argparse._SubParsersAction):
    """Configure a new subparser ."""
    parser = subparsers.add_parser(
        "fine-tune",
        help="Fine-Tune a model on a specific dataset.",
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
        "--epochs",
        metavar="EPOCHS",
        type=int,
        default=10,
        help="Number of epochs for training.",
    )
    parser.add_argument(
        "--save-every",
        metavar="SAVE_EVERY",
        type=int,
        default=100,
        help="Save model every SAVE_EVERY steps.",
    )
    parser.add_argument(
        "--lr",
        metavar="LR",
        type=float,
        default=1e-5,
        help="Learning rate for training.",
    )
    parser.add_argument(
        "--max-patience",
        metavar="MAX_PATIENCE",
        type=int,
        default=2,
        help="Maximum patience for early stopping.",
    )
    parser.add_argument(
        "-r",
        "--rank",
        metavar="R",
        type=int,
        default=32,
        help="Rank for LoRA.",
    )
    parser.add_argument(
        "--lora-alpha",
        metavar="LORA_ALPHA",
        type=int,
        default=64,
        help="Alpha value for LoRA.",
    )

    parser.add_argument(
        "--instruction",
        metavar="INSTR",
        type=str,
        default="Answer using as few words as possible.",
        help="Instruction used for generation.",
    )
    parser.add_argument(
        "--open-ended-questions",
        "-oq",
        action="store_true",
        help="Use open ended questions.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use only 100 questions to check the answer of the model.",
    )
    parser.add_argument(
        "--add-pointing-coords",
        action="store_true",
        help="Add coordinates to the answers.",
    )
    parser.add_argument(
        "--add-distractors-coords",
        action="store_true",
        help="Add coordinates of distractors to the answers.",
    )
    parser.add_argument(
        "--add-xs",
        action="store_true",
        help="Add Xs instead of the coordinates.",
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of runs")

    parser.set_defaults(func=main)


def main(args):
    # removes the function from the args for serialization
    del args.func

    model_name, Model, Collator, lora_targets = MAP_MODELS_TRAIN[args.model_name]

    for i in tqdm(range(args.runs), unit="run", desc="Executing runs"):
        seed = args.seed
        if args.runs > 1:
            seed = i
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # get the model and processor based on the model name
        Processor = partial(AutoProcessor.from_pretrained, model_name, use_fast=True)
        if "qwen2" in args.model_name:
            # as declared in the paper
            processor = Processor(
                min_pixels=100 * 28 * 28,
                max_pixels=16384 * 28 * 28,
            )
        elif args.model_name in ["molmo-o-7b", "llava-onevision-8b"]:
            processor = Processor(trust_remote_code=True)
        else:
            processor = Processor()

        output_folder = args.out_dir
        if args.runs > 1:
            output_folder = os.path.join(output_folder, f"run_{i}")

        if not os.path.exists(os.path.join(output_folder, "training_args.json")):
            # create output folder
            os.makedirs(output_folder, exist_ok=True)
            save_training_params(args, output_folder)
        else:
            # load training parameters from existing folder
            print(f"Experiment '{args.experiment_name}' already exists.")
            print(f"Loading training parameters from {output_folder}.")
            args = load_training_params(output_folder)

        criterion = CrossEntropyLoss(ignore_index=-100, reduction="sum")

        model = Model(
            model_name,
            device_map="auto" if args.parallel else args.device,
        )

        checkpointer = Checkpointer(args)
        early_stopping = checkpointer.checkpoint.early_stopping
        if os.path.exists(os.path.join(output_folder, "checkpoint")):
            if "cuda" in args.device:
                checkpoint, model = checkpointer.load_checkpoint(model, output_folder)
            else:
                assert args.device == "cpu"
                checkpoint, model = checkpointer.load_checkpoint(
                    model, output_folder, args.device  # type: ignore
                )
            early_stopping = checkpoint.early_stopping
            start_epoch = checkpoint.epoch
        else:
            # Initialize training for the first time
            start_epoch = 0

            # default configuration from https://huggingface.co/docs/peft/en/developer_guides/quantization
            config = LoraConfig(
                r=args.rank,
                lora_alpha=args.lora_alpha,
                init_lora_weights="gaussian",
                target_modules=lora_targets,
            )

            model = get_peft_model(model, config)

        model.print_trainable_parameters()

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)  # type: ignore

        if checkpointer.checkpoint.optimizer is not None:
            optimizer.load_state_dict(checkpointer.checkpoint.optimizer)

        with open(os.path.join(args.data_folder, "train_ids.json")) as f:
            train_ids = json.load(f)
        with open(os.path.join(args.data_folder, "valid_ids.json")) as f:
            valid_ids = json.load(f)

        train_ds = CIVETDataset(
            args.data_folder,
            predicted_samples=set(),
            debug=args.debug,
            open_ended_questions=args.open_ended_questions,
            ids_to_process=train_ids,
            add_coords=args.add_pointing_coords,
            add_distractors_coords=args.add_distractors_coords,
            add_xs=args.add_xs,
        )
        valid_ds = CIVETDataset(
            args.data_folder,
            predicted_samples=set(),
            debug=args.debug,
            open_ended_questions=args.open_ended_questions,
            ids_to_process=valid_ids,
            add_coords=args.add_pointing_coords,
            add_distractors_coords=args.add_distractors_coords,
            add_xs=args.add_xs,
        )

        collator = Collator(
            processor=processor,  # type: ignore
            instruction=args.instruction,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collator,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(seed),
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collator,
        )

        if resume_training(start_epoch, args, early_stopping):
            train(
                args,
                model,  # type: ignore
                train_loader,  # type: ignore
                valid_loader,
                criterion,
                optimizer,
                output_folder,
                checkpointer,
                processor.tokenizer,
            )

        # Free memory
        del model
        torch.cuda.empty_cache()
        gc.collect()
