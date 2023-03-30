from abc import ABC, abstractmethod


class Dataset(ABC):
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    @abstractmethod
    def get_tokenized_dataset(self):
        pass

    @abstractmethod
    def preprocess(self, examples):
        pass

    @abstractmethod
    def compute_metrics(self):
        pass
