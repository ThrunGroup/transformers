from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import accuracy_score
import time
from typing import List
from data.get_data import get_data


def inference_pipeline(
        model_names: List[str] = ["bert-base-cased"],
        datasets: List[str] = ["sst2"]
):
    # Set up the tokenizer and models
    models = {}
    for model_name in model_names:
        models[model_name] = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Set up the pipeline for sentiment analysis
    text_classification_pipelines = {
        model_name: pipeline("text-classification", model=model, tokenizer=AutoTokenizer.from_pretrained(model_name))
        for model_name, model in models.items()
    }

    # Benchmark the inference time for each model and dataset
    for dataset_name in datasets:
        print(f"Dataset: ", dataset_name)
        inputs, labels = get_data(dataset_name)
        for model_name, model in text_classification_pipelines.items():
            print(f"Model: ", model_name)
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            inference_time = end_time - start_time

            accuracy = accuracy_score(labels, outputs)
            print(f" Inference time: {inference_time:.4f} seconds")
            print(f" Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    inference_pipeline(model_names=["google/bigbird-roberta-base"])
