from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset

import time

from typing import List


def inference_pipeline(model_names: List[str] = ["bert-base-cased"], num_experiments: int = 1):
    # Set up the tokenizer and models
    models = {}
    for model_name in model_names:
        models[model_name] = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Set up the pipeline for sentiment analysis
    text_classification_pipelines = {
        model_name: pipeline("text-classification", model=model, tokenizer=AutoTokenizer.from_pretrained(model_name))
        for model_name, model in models.items()
    }

    # Set up sequence lengths to test
    # sequence_lengths = [512, 1024]
    # dataset = load_dataset('glue', 'cola', split='test')
    # print(f"Number of test samples is {len(dataset)}")
    # dataset = dataset['sentence']
    sequence_lengths = [500, 1000]
    # Benchmark the inference time for each model and sequence length
    for model_name, text_classification_pipeline in text_classification_pipelines.items():
        # start_time = time.time()
        # output = text_classification_pipeline(dataset)
        # end_time = time.time()
        # inference_time = end_time - start_time
        # # print(output)
        # print(f" Inference time: {inference_time:.4f} seconds")
        #
        # print(f"Model: {model_name}")
        for sequence_length in sequence_lengths:
            input_text = "I'm 3 years older than my brother. My brother is 2 years old. What is my age" * (
                sequence_length // 6
            )
            inference_time = 0

            for idx in range(num_experiments):
                start_time = time.time()
                output = text_classification_pipeline(input_text)
                end_time = time.time()
                inference_time += end_time - start_time
            inference_time /= num_experiments
            print(f"Sequence length: {sequence_length}, Inference time: {inference_time:.4f} seconds")


if __name__ == "__main__":
    inference_pipeline(model_names=["google/bigbird-roberta-base"])



