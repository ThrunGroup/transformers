from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import Trainer
from datasets import load_metric
import time
import numpy as np
from typing import List
from data.get_data import get_data


def compute_metrics(eval_pred):
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": accuracy, "f1": f1}


def inference_pipeline(
        model_names: List[str] = ["bert-base-cased"],
        datasets: List[str] = ["sst2"]
):
    # Set up the tokenizer and models
    models = {}
    for model_name in model_names:
        models[model_name] = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # Set up the pipeline for sentiment analysis
    sentiment_analysis_pipelines = {
        model_name: pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        for model_name, model in models.items()
    }

    # Benchmark the inference time for each model and dataset
    for dataset_name in datasets:
        print(f"Dataset: {dataset_name}")
        tokenized = get_data(dataset_name, tokenizer=tokenizer)

        print(tokenized)

        for model_name, model in sentiment_analysis_pipelines.items():
            print(f"Model: {model_name}")

            evaluator = Trainer(
                model=model,
                eval_dataset=tokenized,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
            )

            evaluator.evaluate()

            # for sequence_length in sequence_lengths:
            #     input_text = "I'm 3 years older than my brother. My brother is 2 years old. What is my age" * (
            #             sequence_length // 6
            #     )
            #     inference_time = 0
            #
            #     for idx in range(num_experiments):
            #         start_time = time.time()
            #         output = sentiment_analysis_pipeline(input_text)
            #         end_time = time.time()
            #         inference_time += end_time - start_time
            #     inference_time /= num_experiments
            #     print(f"Sequence length: {sequence_length}, Inference time: {inference_time:.4f} seconds")


if __name__ == "__main__":
    inference_pipeline(model_names=["google/bigbird-roberta-base"])
