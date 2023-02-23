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

    dataset = load_dataset('glue', 'cola', split='test')  # Hard-coded
    print(f"Number of test samples is {len(dataset)}")
    dataset = dataset['sentence'][:500]

    for model_name, text_classification_pipeline in text_classification_pipelines.items():
        start_time = time.time()
        output = text_classification_pipeline(dataset)
        end_time = time.time()
        inference_time = end_time - start_time
        print(f" Inference time: {inference_time:.4f} seconds")


if __name__ == "__main__":
    inference_pipeline(model_names=["google/bigbird-roberta-base"])



