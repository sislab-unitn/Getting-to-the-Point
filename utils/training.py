import json
import math
import os
from argparse import Namespace
from typing import Iterator, List, Optional, Tuple

import torch
from peft.peft_model import PeftModel
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, PreTrainedTokenizer


def compute_nll_and_ppl(
    losses: List[float], unmasked_tokens: int
) -> Tuple[float, float]:
    nll = sum(losses) / unmasked_tokens
    ppl = math.exp(nll)
    return nll, ppl


def evaluate(
    args,
    model: AutoModelForCausalLM,
    dataloader: DataLoader,
    criterion: CrossEntropyLoss,
    tokenizer: PreTrainedTokenizer,
    is_test: bool = False,
) -> Tuple[float, float]:

    losses = []
    model.eval()  #  type: ignore
    unmasked_tokens = 0
    with torch.no_grad():
        for input_ids, labels in (pbar := tqdm(dataloader, desc="Evaluating")):
            if not isinstance(input_ids, dict):
                input_ids = input_ids.to(args.device)
                in_ids = input_ids.input_ids
            else:
                # molmo processor returns a dict
                input_ids = {k: v.to(args.device) for k, v in input_ids.items()}
                in_ids = input_ids["input_ids"]
            labels = labels.to(args.device)

            with torch.no_grad():
                outputs = model(**input_ids)  # type: ignore

                labels_shifted = torch.full(in_ids.shape, tokenizer.pad_token_id)  # type: ignore

                labels_shifted[labels_shifted == tokenizer.pad_token_id] = -100

                # labels = input tokens shifted left
                labels_shifted[:, :-1] = labels[:, 1:].detach().clone()

                losses.append(
                    criterion(
                        outputs.logits.permute(0, 2, 1), labels_shifted.to(args.device)
                    ).item()
                )
                unmasked_tokens += (labels_shifted != -100).sum().item()
                _, ppl = compute_nll_and_ppl(losses, unmasked_tokens)

            pbar.set_postfix(
                {
                    f"{'Test' if is_test else 'Valid'} PPL": ppl,
                }
            )

    return compute_nll_and_ppl(losses, unmasked_tokens)


def save_training_params(args: Namespace, output_folder: str) -> None:
    with open(os.path.join(output_folder, "training_params.json"), "w") as f:
        json.dump(vars(args), f, indent=4)


def load_training_params(output_folder: str) -> Namespace:
    with open(os.path.join(output_folder, "training_params.json"), "r") as f:
        return Namespace(**json.load(f))


class EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience
        self.counter = 0
        self.best_ppl = float("inf")
        self.stopped = False

    def should_stop(self, current_ppl: float) -> bool:
        if current_ppl < self.best_ppl:
            self.best_ppl = current_ppl
            self.counter = 0
        else:
            self.counter += 1

        # save the counter condition for the checkpointer
        self.stopped = self.counter >= self.patience
        return self.stopped


class Checkpoint:
    def __init__(self, args: Namespace):
        self.epoch = 0
        self.step = 0
        self.optimizer: Optional[dict] = None
        self.early_stopping = EarlyStopping(args.max_patience)
        self.train_stats = []
        self.losses = []
        self.unmasked_tokens = 0


class Checkpointer:
    def __init__(self, args: Namespace):
        self.checkpoint = Checkpoint(args)

    def update_checkpoint(
        self,
        model: PeftModel,
        optimizer: torch.optim.Optimizer,  # type: ignore
        step: int,
        losses: List[float],
        unmasked_tokens: int,
        output_folder: str,
        train_stats: Optional[List[dict]] = None,
        early_stopping: Optional[EarlyStopping] = None,
        epoch: Optional[int] = None,
    ):
        model.save_pretrained(os.path.join(output_folder, "checkpoint"))
        self.checkpoint.optimizer = optimizer.state_dict()
        if epoch is not None:
            self.checkpoint.epoch = epoch
        self.checkpoint.step = step
        self.checkpoint.losses = losses
        self.checkpoint.unmasked_tokens = unmasked_tokens
        if train_stats is not None:
            self.checkpoint.train_stats = train_stats
        if early_stopping is not None:
            self.checkpoint.early_stopping = early_stopping
        torch.save(
            self.checkpoint, os.path.join(output_folder, "checkpoint", "checkpoint.pt")
        )

    def load_checkpoint(
        self,
        model: AutoModelForCausalLM,
        output_folder: str,
        device: Optional[torch.device] = None,
    ) -> Tuple[Checkpoint, PeftModel]:
        model = PeftModel.from_pretrained(
            model, os.path.join(output_folder, "checkpoint"), is_trainable=True  # type: ignore
        )
        if device is not None:
            self.checkpoint = torch.load(
                os.path.join(output_folder, "checkpoint", "checkpoint.pt"),
                weights_only=False,
                map_location=device,
            )
        else:
            self.checkpoint = torch.load(
                os.path.join(output_folder, "checkpoint", "checkpoint.pt"),
                weights_only=False,
            )
        return self.checkpoint, model  # type: ignore


def resume_training(epoch: int, args: Namespace, early_stopping: EarlyStopping) -> bool:
    if epoch >= args.epochs:
        print(f"Reached maximum number of epochs {epoch}.")
        return False
    if early_stopping.stopped:
        print(f"Early stopping at epoch {epoch}")
        return False
    return True


def train_one_epoch(
    args,
    model: PeftModel,
    optimizer: torch.optim.Optimizer,  # type: ignore
    train_iterator: Iterator,
    dataloader: DataLoader,
    start_step: int,
    steps_so_far: int,
    checkpointer: Checkpointer,
    output_folder: str,
) -> Tuple[Tuple[float, float], int]:

    model.train()  # type: ignore
    losses = checkpointer.checkpoint.losses
    unmasked_tokens = checkpointer.checkpoint.unmasked_tokens

    # Resume training for the current step
    for _ in range(start_step):
        next(train_iterator)

    with tqdm(dataloader, desc="Training") as pbar:
        pbar.total = len(dataloader)
        pbar.n = start_step
        pbar.refresh()
        for step, (input_ids, labels) in enumerate(train_iterator, start=start_step):
            if not isinstance(input_ids, dict):
                input_ids = input_ids.to(args.device)
            else:
                # molmo processor returns a dict
                input_ids = {k: v.to(args.device) for k, v in input_ids.items()}
            labels = labels.to(args.device)

            optimizer.zero_grad()

            outputs = model(**input_ids, labels=labels)

            loss = outputs.loss

            # store the original loss
            losses.append(loss.item())
            # compute the number of unmasked tokens for this batch
            batch_unmasked_tokens = (labels != -100).sum().item()
            # normalize the loss
            loss /= batch_unmasked_tokens
            loss.backward()
            optimizer.step()

            unmasked_tokens += batch_unmasked_tokens
            _, ppl = compute_nll_and_ppl(losses, unmasked_tokens)
            pbar.set_postfix(
                {
                    "Train PPL": ppl,
                }
            )
            pbar.update(1)
            steps_so_far += 1

            if steps_so_far % args.save_every == 0:
                checkpointer.update_checkpoint(
                    model,
                    optimizer,
                    step + 1,  # We finished the current step, so we increment the step
                    losses,
                    unmasked_tokens,
                    output_folder,
                )

    return compute_nll_and_ppl(losses, unmasked_tokens), steps_so_far


def train(
    args: Namespace,
    model: PeftModel,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    criterion: CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,  # type: ignore
    output_folder: str,
    checkpointer: Checkpointer,
    tokenizer: PreTrainedTokenizer,
) -> None:

    start_epoch = checkpointer.checkpoint.epoch
    start_step = checkpointer.checkpoint.step
    train_stats = checkpointer.checkpoint.train_stats
    early_stopping = checkpointer.checkpoint.early_stopping

    steps_so_far = len(train_loader) * start_epoch + start_step

    # Resume training for the current epoch
    for _ in range(start_epoch):
        iter(train_loader)

    best_ppl = early_stopping.best_ppl
    for epoch in trange(
        start_epoch,
        args.epochs,
        desc="Epochs",
        initial=start_epoch,
        total=args.epochs,
    ):
        train_iterator = iter(train_loader)
        (train_nnl, train_ppl), steps_so_far = train_one_epoch(
            args,
            model=model,
            optimizer=optimizer,
            train_iterator=train_iterator,
            dataloader=train_loader,
            start_step=start_step,
            steps_so_far=steps_so_far,
            checkpointer=checkpointer,
            output_folder=output_folder,
        )
        valid_nll, valid_ppl = evaluate(args, model, valid_loader, criterion, tokenizer=tokenizer)  # type: ignore

        # Save best model
        if valid_ppl < best_ppl:
            best_ppl = valid_ppl
            model.save_pretrained(os.path.join(output_folder, "best_model"))

        # Save the results for the checkpointer
        should_stop = early_stopping.should_stop(valid_ppl)

        train_stats.append(
            {
                "Epoch": epoch,
                "Train NLL": train_nnl,
                "Train PPL": train_ppl,
                "Valid NLL": valid_nll,
                "Valid PPL": valid_ppl,
                "Patience": early_stopping.patience - early_stopping.counter,
            }
        )

        # Update the checkpointer
        checkpointer.update_checkpoint(
            model=model,
            optimizer=optimizer,
            step=0,  # We finished the epoch, so we reset the step
            losses=[],  # We finished the epoch, so we reset the losses
            unmasked_tokens=0,  # We finished the epoch, so we reset the unmasked tokens
            output_folder=output_folder,
            train_stats=train_stats,
            early_stopping=early_stopping,
            epoch=epoch + 1,  # We finished the current epoch, so we increment the epoch
        )

        start_step = 0

        with open(os.path.join(output_folder, "train_stats.json"), "w") as f:
            json.dump(train_stats, f, indent=4)

        # Early stopping
        if should_stop:
            break
