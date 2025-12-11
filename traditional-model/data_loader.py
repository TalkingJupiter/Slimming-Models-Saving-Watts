from datasets import load_dataset


def load_sharded_dataset(shards_file: str, split: str = "train"):
    dataset = load_dataset("json", data_files=shards_file, split=split)
    return dataset
