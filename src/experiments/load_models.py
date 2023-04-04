from transformers import (
    AutoTokenizer,
    TransfoXLLMHeadModel,
    BloomForCausalLM,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OPTForCausalLM,
)
import torch
import os
import ast

from utils.constants import TRANSFORMER_XL, GPT2, GPT2_LARGE, NUM_BLOCKS_GPT2, SVD, OPT, BLOOM
from utils.parse_string import get_model_type, parse_string, string_to_dict
from accelerators.apply_accelerator import apply_accelerator


def get_naive_model_and_tokenizer(model_name: str):
    """
    Get the naive model and tokenizer from HuggingFace

    :param model_name: Name of the model to get
    :return: Model and tokenizer
    """
    if model_name == TRANSFORMER_XL:
        pretrained_model = "transfo-xl-wt103"
        model = TransfoXLLMHeadModel.from_pretrained(pretrained_model, device_map="auto",)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    elif GPT2 in model_name:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id, device_map="auto",)
    elif OPT in model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            "facebook/" + model_name, use_fast=False
        )  # use_fast = False to get correct tokenizer
        model = OPTForCausalLM.from_pretrained("facebook/" + model_name, device_map="auto",)
    elif BLOOM in model_name:
        tokenizer = AutoTokenizer.from_pretrained("bigscience/" + model_name, use_fast=False)
        model = BloomForCausalLM.from_pretrained("bigscience/" + model_name, device_map="auto",)
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

    # Get the path to the model checkpoint
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_dir, "checkpoints", model_name)
    last_checkpoint = os.listdir(checkpoint_path)[-1]
    model_path = os.path.join(checkpoint_path, last_checkpoint)

    # Load the model
    if model_type == GPT2:
        tokenizer = GPT2Tokenizer.from_pretrained(GPT2)
        model = GPT2LMHeadModel.from_pretrained(model_path, pad_token_id=tokenizer.eos_token_id)
    elif model_type == TRANSFORMER_XL:
        model = TransfoXLLMHeadModel.from_pretrained(model_path)
    else:
        assert False, f"{model_name} is not valid"

    # Extract key information from the model name to apply the right accelerator to the model
    # For example, given "gpt2_SVD_{'k': 10}_accelerated_0-11_froze_0-10_dataset_billsum_trained_accelerated_layers"
    # extract: accelerator = SVD, accelerate_args = {'k': 10}, layers_to_accelerate = 0-11
    model_keywords = model_name.split("_")

    accelerator_type = model_keywords[1]

    str_accelerator_args = model_keywords[2]
    accelerator_args = string_to_dict(str_accelerator_args)

    # str_layers_to_accelerate = model_keywords[4]
    # if str_layers_to_accelerate != "None":
    #     layers_to_accelerate = parse_string(str_layers_to_accelerate)
    #
    #     apply_accelerator(model_type, model, layers_to_accelerate, accelerator_type,
    #                       **accelerator_args)

    # Load the checkpoint
    checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
    else:
        assert False, f"{model_name} could not be found in the checkpoint folder"

    print(f"Successfully loaded {model_name}")

    return model


def list_checkpoint_models():
    """
    List the models found in the checkpoint directory
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_dir, "checkpoints")
    return os.listdir(checkpoint_path)


if __name__ == "__main__":
    model_name = "gpt2_SVD_{'k': 10}_accelerated_11_froze_0-10_dataset_billsum_trained_accelerated_layers"
    new_model = load_model(model_name)
