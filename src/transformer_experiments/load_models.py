from transformers import AutoTokenizer, TransfoXLLMHeadModel, GPT2LMHeadModel, GPT2Tokenizer
import time
from typing import List

from utils.constants import TRANSFORMER_XL, SVD, PCA, GPT2, GPT2_LARGE
from accelerators.apply_accelerator import apply_accelerator
from datasets import load_dataset


def get_naive_model_and_tokenizer(model_name: str):
    if model_name == TRANSFORMER_XL:
        pretrained_model = "transfo-xl-wt103"
        model = TransfoXLLMHeadModel.from_pretrained(pretrained_model)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    elif GPT2 in model_name:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)
    else:
        assert False, "No such model"

    return model, tokenizer
