from transformers import AutoTokenizer, AutoModel, pipeline, TransfoXLLMHeadModel
import time
from typing import List


def get_model_tokenizer(model_name: str):
    if model_name == "transformer_xl":
        pretrained_model = "transfo-xl-wt103"
        model = TransfoXLLMHeadModel.from_pretrained(pretrained_model)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        return model, tokenizer

    assert False, "No such model"


def inference_pipeline(
        model_names: List[str] = ["transformer_xl"],
):
    # Set up the tokenizer and models
    models = {}
    tokenizers = {}
    for model_name in model_names:
        model, tokenizer = get_model_tokenizer(model_name)
        models[model_name] = model
        tokenizers[model_name] = tokenizer

    # Benchmark the inference time for each model
    # Encode input sequence
    for model_name, model in models.items():
        print(f"Model: ", model_name)
        tokenizer = tokenizers[model_name]
        input_ids = tokenizer.encode("Once upon a time", return_tensors="pt")  # Encode input sequence
        start_time = time.time()
        output_ids = model.generate(input_ids, max_length=50)  # Generate text
        end_time = time.time()
        inference_time = end_time - start_time

        # Decode output sequence
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print("Model output: ", output_text)
        print(f" Inference time: {inference_time:.4f} seconds")


if __name__ == "__main__":
    inference_pipeline(model_names=["transformer_xl"])
