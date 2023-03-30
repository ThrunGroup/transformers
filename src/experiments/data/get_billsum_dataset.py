from transformers import DataCollatorForSeq2Seq
import evaluate
import numpy as np
from datasets import load_dataset

from .dataset import Dataset


class Billsum(Dataset):
    def __init__(self, tokenizer, model, metrics: str = "rouge", prefix: str = "summarize: "):
        """
        :param tokenizer: Tokenizer to tokenize the inputs
        :param model: Model to finetune / evaluate
        :param metrics: Metrics to evaluate the model
        :param prefix: Prefix to the inputs to start the summarization task
        """
        super().__init__(tokenizer, model)
        self.metrics = evaluate.load(metrics)  # Load the metrics to be used by compute_metrics
        self.prefix = prefix

    def get_tokenized_dataset(self):
        """
        Load and build a simple summarization dataset
        :return: tokenized dataset, data collator, and compute_metrics function
        """
        dataset = load_dataset("billsum", split="train[:100]")
        dataset = dataset.train_test_split(test_size=0.2)

        tokenized_dataset = dataset.map(self.preprocess, batched=True)

        return tokenized_dataset["train"], tokenized_dataset["test"]

    def get_data_collator(self):
        """
        Load the data collator that is responsible for taking in batches of examples
        and converting them into a format that can be consumed by the model.
        """
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)
        return data_collator

    def get_compute_metrics(self):
        """
        Load the compute_metrics function that is used by the HuggingFace Trainer to
        evaluate the model.
        :return:
        """
        return self.compute_metrics

    def preprocess(self, examples):
        inputs = [self.prefix + doc for doc in examples["text"]]
        model_inputs = self.tokenizer(inputs, max_length=128, truncation=True)

        # TODO: decrease max_length of labels
        #       (but then it raises ValueError: Expected input batch_size (2032) to match target batch_size (496).)
        labels = self.tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

        # model_inputs["input_ids"] = [ids[:-1] for ids in model_inputs["input_ids"]]
        # model_inputs["attention_mask"] = [mask[:-1] for mask in model_inputs["attention_mask"]]
        # # model_inputs["labels"] = labels["input_ids"]
        # model_inputs["decoder_attention_mask"] = labels["attention_mask"]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = self.metrics.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}
