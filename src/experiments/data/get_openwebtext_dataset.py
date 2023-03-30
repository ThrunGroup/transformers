from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset

from .dataset import Dataset


class OpenWebText(Dataset):
    def __init__(self, tokenizer, model):
        """
        :param tokenizer: Tokenizer to tokenize the inputs
        :param model: Model to finetune / evaluate
        :param metrics: Metrics to evaluate the model
        :param prefix: Prefix to the inputs to start the summarization task
        """
        super().__init__(tokenizer, model)
        # self.tokenizer.pad_token = tokenizer.mask_toke
        self.tokenizer.pad_token = tokenizer.eos_token

    def get_tokenized_dataset(self):
        """
        Load and build the openwebtext dataset which is already preprocessed.
        :return: dataset
        """
        dataset = load_dataset("stas/openwebtext-10k", split="train[:10]")
        dataset = dataset.train_test_split(test_size=0.2)
        tokenized_dataset = dataset.map(self.preprocess, batched=True)
        print(tokenized_dataset)

        # train_dataset = TextDataset(tokenizer=self.tokenizer, file_path=train_dataset['text'], block_size=128)
        # test_dataset = TextDataset(tokenizer=self.tokenizer, file_path=test_dataset['text'], block_size=128)

        return tokenized_dataset["train"], tokenized_dataset["test"]

    def preprocess(self, examples):
        inputs = self.tokenizer(examples["text"], max_length=1024, truncation=True)

        return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}

    def get_data_collator(self):
        """
        Load the data collator that is responsible for taking in batches of examples
        and converting them into a format that can be consumed by the model.
        """
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False, mlm_probability=0.15
        )
        return data_collator

    def get_compute_metrics(self):
        """
        Load the compute_metrics function that is used by the HuggingFace Trainer to
        evaluate the model.
        :return:
        """
        return self.compute_metrics()

    def compute_metrics(self):
        return None
