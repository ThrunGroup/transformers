from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding


def get_data(dataset_name: str, tokenizer=None):
    if dataset_name == "sst2":
        dataset = load_dataset('glue', 'sst2')
    else:
        raise NotImplementedError(f"{dataset_name} is not available")

    test_dataset = dataset["test"].shuffle(seed=42).select([i for i in list(range(300))])

    print(test_dataset)

    def _preprocess_function(examples):
        return tokenizer(examples["sentence"], truncation=True)

    # tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
    tokenized = test_dataset.map(_preprocess_function, batched=True)

    print(tokenized)

    # Convert the samples to PyTorch tensors and concatenate them with the correct amount of padding
    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return tokenized


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
get_data("sst2", tokenizer)
