from transformers import AutoTokenizer, TransfoXLLMHeadModel
import time
from typing import List

from utils.constants import TRANSFORMER_XL, SVD, PCA
from accelerators.apply_accelerator import apply_accelerator


def get_model_tokenizer(model_name: str):
    if model_name == "transformer_xl":
        pretrained_model = "transfo-xl-wt103"
        model = TransfoXLLMHeadModel.from_pretrained(pretrained_model)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        return model, tokenizer

    assert False, "No such model"


def inference_pipeline(
        model_names: List[str] = [TRANSFORMER_XL],
        accelerators: List[str] = [None, "SVD"]
):
    # Set up the tokenizer and models
    models = {}
    tokenizers = {}
    for model_name in model_names:
        model, tokenizer = get_model_tokenizer(model_name)
        models[model_name] = model
        tokenizers[model_name] = tokenizer
        apply_accelerator(model_name, model, SVD, k=4)
        print(model)

    # Benchmark the inference time for each model
    # Encode input sequence
    for model_name, model in models.items():
        print(f"Model: ", model_name)
        tokenizer = tokenizers[model_name]
        input_ids = tokenizer.encode("Once upon a time", return_tensors="pt")  # Encode input sequence
        start_time = time.time()
        output_ids = model.generate(input_ids, max_length=40)  # Generate text
        end_time = time.time()
        inference_time = end_time - start_time

        # Decode output sequence
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print("\tModel output: ", output_text)
        print(f"\tInference time: {inference_time:.4f} seconds")


if __name__ == "__main__":
    inference_pipeline(model_names=[TRANSFORMER_XL])

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
