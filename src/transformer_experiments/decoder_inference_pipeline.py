from transformers import AutoTokenizer, TransfoXLLMHeadModel, GPT2LMHeadModel, GPT2Tokenizer
import time
from typing import List

from utils.constants import TRANSFORMER_XL, SVD, PCA, GPT2, GPT2_LARGE
from accelerators.apply_accelerator import apply_accelerator
from datasets import load_dataset


def get_model_tokenizer(model_name: str):
    if model_name == "transformer_xl":
        pretrained_model = "transfo-xl-wt103"
        model = TransfoXLLMHeadModel.from_pretrained(pretrained_model)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    elif GPT2 in model_name:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)
    else:
        assert False, "No such model"

    return model, tokenizer


def inference_pipeline(
    model_names: List[str] = [GPT2], accelerators: List[str] = [None, "SVD"],
):
    # Set up the tokenizer and models
    models = {}
    tokenizers = {}
    for accelerator in accelerators:
        for model_name in model_names:
            model, tokenizer = get_model_tokenizer(model_name)
            apply_accelerator(model_name, model, accelerator, k=200)
            new_model_name = model_name + f"+ {accelerator}"
            models[new_model_name] = model
            tokenizers[new_model_name] = tokenizer

    # Benchmark the inference time for each model
    # Encode input sequence
    # dataset = load_dataset("cbt", "CN")
    for model_name, model in models.items():
        print(f"Model: ", model_name)
        tokenizer = tokenizers[model_name]
        input_ids = tokenizer.encode(
            "Once upon a week, ",
            return_tensors="pt",
        )  # Encode input sequence
        start_time = time.time()
        output_ids = model.generate(input_ids, max_length=50)  # Generate text
        end_time = time.time()
        inference_time = end_time - start_time

        # Decode output sequence
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print("\tModel output: ", output_text)
        print(f"\tInference time: {inference_time:.4f} seconds")


if __name__ == "__main__":
    inference_pipeline(model_names=[GPT2])

"""
SVD 400, max 30
    Model output:  Once upon a time, the "the" the "the" the "the" the "the" the "the" the "the" the
    Inference time: 34.0093 seconds

SVD 400, max 40
    Model output:  Once upon a time, the "the" the "the" the "the" the "the" the "the" the "the" the "the" the "the" the "the
    Inference time: 48.1490 seconds
    
SVD 600, max 40
    Model output:  Once upon a time, the first was the first the first the first was the first the first the first was the first the first the first the first the first the first the first and the first the
    Inference time: 54.8138 seconds
 
Vanilla, max 30
    Model output:  Once upon a time, would be able to take advantage of the opportunity to take advantage of the opportunity to take advantage of the opportunity to take advantage of
    Inference time: 36.2548 seconds

Vanilla, max 40
    Model output:  Once upon a time, would be able to take advantage of the opportunity to take advantage of the opportunity to take advantage of the opportunity to take advantage of the opportunity to take advantage of the opportunity to take
    Inference time: 49.8048 seconds
    
PCA 64, max 40
    Model output:  Once upon a time the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the
    Inference time: 44.4036 seconds
"""
