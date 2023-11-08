import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

def expect_batch_inputs():
    sequence = "I've been waiting for this session my whole life."
    tokens = tokenizer.tokenize(sequence)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor(ids)
    print(input_ids)
    model(input_ids)

def view_tokenizer():
    sequence = "I've been waiting for this session my whole life."
    print(tokenizer(sequence, return_tensors='pt')['input_ids'])

def add_new_dimension():
    sequence = "I've been waiting for this session my whole life."

    tokens = tokenizer.tokenize(sequence)
    ids = tokenizer.convert_tokens_to_ids(tokens)

    input_ids = torch.tensor([ids]) # before: torch.tensor(ids)
    print("Input IDs:", input_ids)

    output = model(input_ids)
    print("Logits:", output.logits)

def fix_padding_issues():
    seq1 = [[200,200,200]]
    seq2 = [[200,200]]
    batched_ids = [
        [200, 200, 200],
        [200, 200, tokenizer.pad_token_id]
    ]
    print(model(torch.tensor(seq1)).logits)
    print(model(torch.tensor(seq2)).logits)
    print(model(torch.tensor(batched_ids)).logits)
    """ 
    The second row should be the same as the logits for the second sentence.
    But no, since attention layers contextualize each token.
    Need to tell those attention layers to ignore the padding tokens
    """
def attention_masks():
    batched_ids = [
        [200, 200, 200],
        [200, 200, tokenizer.pad_token_id],
    ]

    attention_mask = [
        [1, 1, 1],
        [1, 1, 0],
    ]

    outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
    print(outputs.logits)

if __name__ == "__main__":
    # expect_batch_inputs()
    # view_tokenizer()
    # add_new_dimension()
    # fix_padding_issues()
    attention_masks()