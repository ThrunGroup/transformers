from datasets import load_dataset


def get_data(dataset_name: str):
    if dataset_name == "sst2":
        dataset = load_dataset('glue', 'sst2', split='test')
    elif dataset_name == "sst2":
        dataset = load_dataset('glue', 'cola', split='test')
    else:
        raise NotImplementedError(f"{dataset_name} is not available")

    inputs = dataset["sentence"]
    labels = dataset["label"]

    return inputs, labels
