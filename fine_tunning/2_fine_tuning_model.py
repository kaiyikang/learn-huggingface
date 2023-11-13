from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import Trainer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import numpy as np
import evaluate

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

# trainer.train()

##############
# Evaluation #
##############
# predictions = trainer.predict(tokenized_datasets["validation"])
# # 408 being the number of elements 
# print(predictions.predictions.shape, predictions.label_ids.shape)

# # take the index with the maximum value on the second axis
# preds = np.argmax(predictions.predictions, axis=-1)

# print(preds)

# metric = evaluate.load("glue", "mrpc")
# result = metric.compute(predictions=preds, references=predictions.label_ids)

# print(result)

def compute_metrics(eval_preds): # eval_preds = trainer.predict(tokenized_datasets["validation"])
    logits, label_ids = eval_preds

    metric = evaluate.load("glue", "mrpc")
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=label_ids)

print(compute_metrics(eval_preds=trainer.predict(tokenized_datasets["validation"])))