import random
from functools import partial
from typing import List, Optional

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    GenerationConfig,
)
from transformers.modeling_utils import PreTrainedModel

from utils import qwen2_vl, molmo, internvl3


MAP_MODELS = {
    "llava-onevision-8b": (
        "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
        partial(
            AutoModelForCausalLM.from_pretrained,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        ),
        qwen2_vl.GenerationCollator,  # uses the same collator as qwen2-vl-7b
    ),
    "internvl3_5-8b": (
        "OpenGVLab/InternVL3_5-8B-HF",
        partial(
            AutoModelForImageTextToText.from_pretrained,
            dtype=torch.bfloat16,
        ),
        internvl3.GenerationCollator,
    ),
    "qwen2.5-vl-3b": (
        "Qwen/Qwen2.5-VL-3B-Instruct",
        partial(
            AutoModelForImageTextToText.from_pretrained,
            dtype=torch.bfloat16,
        ),
        qwen2_vl.GenerationCollator,
    ),
    "qwen2.5-vl-7b": (
        "Qwen/Qwen2.5-VL-7B-Instruct",
        partial(
            AutoModelForImageTextToText.from_pretrained,
            dtype=torch.bfloat16,
        ),
        qwen2_vl.GenerationCollator,
    ),
    "molmo-o-7b": (
        "allenai/Molmo-7B-O-0924",
        partial(
            AutoModelForCausalLM.from_pretrained,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,  # correct dtype is torch.float32, but performance is negligible
        ),
        molmo.GenerationCollator,
    ),
}

MAP_MODELS_TRAIN = {
    "qwen2.5-vl-3b": (
        "Qwen/Qwen2.5-VL-3B-Instruct",
        partial(
            AutoModelForImageTextToText.from_pretrained,
            dtype=torch.bfloat16,
        ),
        qwen2_vl.TrainCollator,
        ["q_proj", "k_proj", "v_proj", "qkv"],  # qwen (dec, enc)
    ),
    "qwen2.5-vl-7b": (
        "Qwen/Qwen2.5-VL-7B-Instruct",
        partial(
            AutoModelForImageTextToText.from_pretrained,
            dtype=torch.bfloat16,
        ),
        qwen2_vl.TrainCollator,
        ["q_proj", "k_proj", "v_proj", "qkv"],  # qwen (dec, enc)
    ),
    "llava-onevision-8b": (
        "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
        partial(
            AutoModelForCausalLM.from_pretrained,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        ),
        qwen2_vl.TrainCollator,
        ["q_proj", "k_proj", "v_proj", "qkv"],  # qwen (dec), rice (enc)
    ),
    "internvl3_5-8b": (
        "OpenGVLab/InternVL3_5-8B-HF",
        partial(
            AutoModelForImageTextToText.from_pretrained,
            dtype=torch.bfloat16,
        ),
        internvl3.TrainCollator,
        ["q_proj", "k_proj", "v_proj"],  # same for encoder and decoder
    ),
    "molmo-o-7b": (
        "allenai/Molmo-7B-O-0924",
        partial(
            AutoModelForCausalLM.from_pretrained,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,  # correct dtype is torch.float32, but performance is negligible
        ),
        molmo.TrainCollator,
        ["wq", "wk", "wv", "att_proj"],
    ),
}

MAP_MODELS_COORDS = {
    "qwen2.5-vl-3b": (
        "Qwen/Qwen2.5-VL-3B-Instruct",
        partial(
            AutoModelForImageTextToText.from_pretrained,
            dtype=torch.bfloat16,
        ),
        qwen2_vl.CoordinatesCollator,
    ),
    "qwen2.5-vl-7b": (
        "Qwen/Qwen2.5-VL-7B-Instruct",
        partial(
            AutoModelForImageTextToText.from_pretrained,
            dtype=torch.bfloat16,
        ),
        qwen2_vl.CoordinatesCollator,
    ),
    "llava-onevision-8b": (
        "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
        partial(
            AutoModelForCausalLM.from_pretrained,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        ),
        qwen2_vl.CoordinatesCollator,
    ),
    "internvl3_5-8b": (
        "OpenGVLab/InternVL3_5-8B-HF",
        partial(
            AutoModelForImageTextToText.from_pretrained,
            dtype=torch.bfloat16,
        ),
        internvl3.CoordinatesCollator,
    ),
}


def generate(
    input_ids,
    model: PreTrainedModel,
    model_name: str,
    processor,
    max_new_tokens: int,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    temperature: Optional[float] = None,
) -> List[str]:
    if model_name in ["internvl3-8b"] and "input_ids" in input_ids:
        input_ids.to(dtype=model.dtype)

    if model_name == "molmo-o-7b":
        # put the inputs on the correct device and make a batch of size 1
        input_ids = {k: v.to(model.device).unsqueeze(0) for k, v in input_ids.items()}

        # cast the inputs to the right dtype
        input_ids["images"] = input_ids["images"].to(model.dtype)

        assert input_ids["images"].size(0) == 1, "Batch size must be 1 for Molmo-O-7B."

        output = model.generate_from_batch(  # type: ignore
            input_ids,
            GenerationConfig(
                max_new_tokens=max_new_tokens,
                stop_strings="<|endoftext|>",  # suggested on MOLMo huggingface page
                top_k=top_k,
                temperature=temperature,
                top_p=top_p,
            ),
            tokenizer=processor.tokenizer,
        )
    else:
        input_ids.to(model.device)
        with torch.no_grad():
            output = model.generate(
                **input_ids,
                do_sample=False,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                pad_token_id=processor.tokenizer.eos_token_id,  # type: ignore
            )

    if "llava" in model_name or "internvl3" in model_name:
        if "input_ids" in input_ids:
            output = output[:, input_ids.input_ids.size(-1) :]  # type: ignore
    elif "qwen" in model_name:
        if "input_ids" in input_ids:
            output = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(input_ids.input_ids, output)  # type: ignore
            ]
    elif model_name == "molmo-o-7b":
        output = output[:, input_ids["input_ids"].size(1) :]

    if model_name == "molmo-o-7b":
        generated_answers = processor.tokenizer.batch_decode(
            output, skip_special_tokens=True
        )
    else:
        generated_answers = processor.batch_decode(output, skip_special_tokens=True)

    return generated_answers


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
