from transformers import AutoTokenizer, DataCollatorForSeq2Seq
import evaluate
import numpy as np

from datasets import load_dataset


def get_billsum_dataset(tokenizer, model):
    """
    Load and build a simple summarization dataset
    :param tokenizer: Tokenizer to tokenize the inputs
    :param model: Model to finetune / evaluate
    :return: tokenized dataset, data collator, and compute_metrics function
    """
    dataset = load_dataset("billsum", split="ca_test")
    dataset = dataset.train_test_split(test_size=0.2)

    prefix = "summarize: "
    rouge = evaluate.load("rouge")

    def _preprocess_function(examples):
        inputs = [prefix + doc for doc in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

        labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def _compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    tokenized_dataset = dataset.map(_preprocess_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    return tokenized_dataset, data_collator, _compute_metrics
