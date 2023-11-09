"""
MRPC (Microsoft Research Paraphrase Corpus) dataset
- 5,801 pairs of sentences
- a label indicating if they are paraphrases or not
"""

from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="bert-base-uncased")

def load_print_dataset():

    raw_datasets = load_dataset("glue", "mrpc")
    print(raw_datasets)

    raw_train_dataset = raw_datasets['train']
    print(raw_train_dataset[0])
    print(raw_train_dataset.features)
    return raw_datasets

def generate_tokenized_datasets(raw_datasets):
    # tokenize all the first sentences and all the second sentences 
    if False:
        tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
        tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])

    # or like this
    inputs = tokenizer("This is the first sentence.", "This is the second one.")

    # token_type_ids: first or second
    print(inputs)
    print(tokenizer.convert_ids_to_tokens(inputs["input_ids"]))

    if False:
        # returning a dictionary, need enough RAM to store whole dataset
        tokenized_dataset = tokenizer(
            raw_datasets["train"]["sentence1"],
            raw_datasets["train"]["sentence2"],
            padding=True,
            truncation=True,
        )
    
    tokenize_function = lambda example:tokenizer(example["sentence1"], example["sentence2"], truncation=True)
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    print(tokenized_datasets)

    return tokenized_datasets

def generate_dynamic_padding(tokenized_datasets):

    from transformers import DataCollatorWithPadding
    # to know which padding token to use, and whether the model expects padding to be on the left or on the right of the inputs
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    samples = tokenized_datasets["train"][:8]
    samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
    # look at the lengths of each entry
    print([len(x) for x in samples["input_ids"]])

    batch = data_collator(samples)
    print({k: v.shape for k, v in batch.items()})


if __name__ == "__main__":
    raw_datasets = load_print_dataset()
    tokenized_datasets = generate_tokenized_datasets(raw_datasets=raw_datasets)
    generate_dynamic_padding(tokenized_datasets=tokenized_datasets)