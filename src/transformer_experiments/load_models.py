from transformers import AutoTokenizer, TransfoXLLMHeadModel, GPT2LMHeadModel, GPT2Tokenizer
import torch
import os

from utils.constants import TRANSFORMER_XL, GPT2, GPT2_LARGE
from utils.parse_string import get_model_type


def get_naive_model_and_tokenizer(model_name: str):
    """
    Get the naive model and tokenizer from HuggingFace

    :param model_name: Name of the model to get
    :return: Model and tokenizer
    """
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


def load_model(model_name: str):
    """
    Load the fine-tuned model

    :param model_name: Name of the model to load
    :return: Loaded model
    """
    model_type = get_model_type(model_name)

    # Load the model
    if model_type == GPT2:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)
    elif model_type == TRANSFORMER_XL:
        pretrained_model = "transfo-xl-wt103"
        model = TransfoXLLMHeadModel.from_pretrained(pretrained_model)
    else:
        assert False, f"{model_name} is not valid"

    # Load the checkpoint
    checkpoint_path = f'./checkpoints/{model_name}'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        assert False, f"{model_name} could not be found in the checkpoint folder"

    return model
