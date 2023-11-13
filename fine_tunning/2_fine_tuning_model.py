from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import Trainer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

############
# Training #
############
"""
TrainingArguments class contains all the hyperparameters the Trainer will use for training and evaluation.
Only argument: directory where the trained model will be saved, as well as the checkpoints along the way.
"""
training_args = TrainingArguments("test-trainer")

# Get Model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# Get Trainer
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    # data_collator=data_collator,  # can ignore, since it is DataCollatorWithPadding(tokenizer=tokenizer)
    tokenizer=tokenizer,
)

trainer.train()

# data = get_data()
# print(get_training_args())
