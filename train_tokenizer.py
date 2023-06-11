from datasets import load_dataset
from transformers import AutoTokenizer

def get_training_corpus():
    dataset = raw_datasets["train"]
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield str(samples["text"])

# load train dataset
raw_datasets = load_dataset("csv", data_files="train.csv")
old_tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
# train tokenizer on own data
new_tokenizer = old_tokenizer.train_new_from_iterator(get_training_corpus(), 5000, min_frequence = 2)
new_tokenizer.save_pretrained("own-tokenizer")