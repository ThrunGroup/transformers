from datasets import load_dataset


def get_data(dataset_name: str, num_data: int):
    """
    Load a dataset from HuggingFace

    :param dataset_name: name of the dataset to load (sst2, cola, etc.)
    :param num_data: number of data to load
    :return: inputs and labels of the dataset
    """
    if dataset_name == "sst2":
        dataset = load_dataset('glue', 'sst2', split='test')
    elif dataset_name == "cola":
        dataset = load_dataset('glue', 'cola', split='test')
    else:
        raise NotImplementedError(f"{dataset_name} is not available")

    inputs = dataset["sentence"]
    labels = dataset["label"]

    if num_data is not None:
        num_original_data = len(inputs)
        assert num_original_data >= num_data, \
            f"Number of data to load ({num_data}) must be less than {num_original_data} for {dataset_name}"
        inputs = inputs[:num_data]
        labels = labels[:num_data]

    return inputs, labels


if __name__ == "__main__":
    inputs, labels = get_data("sst2", 100)
    print(inputs)
