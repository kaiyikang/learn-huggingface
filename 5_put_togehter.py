from transformers import AutoTokenizer


checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequences = ["I've been waiting for this session course my whole life.", "Ich bin auch!" ]


def get_model_inputs():

    padding_mode = "truncation"

    if padding_mode == "longest":
        model_inputs = tokenizer(sequences, padding=padding_mode)
    elif padding_mode == "max_length":
        model_inputs = tokenizer(sequences, padding=padding_mode)
    elif padding_mode.isdigit():
        model_inputs = tokenizer(sequences, padding="max_length", max_length=int(padding_mode)) 
    elif padding_mode == "truncation":
        # Will truncate the sequences that are longer than the model max length -> BERT 512
        model_inputs = tokenizer(sequences, truncation=True)

    print(model_inputs)

def special_tokens():
    sequence = "I've been waiting for the session my whole life."

    model_inputs = tokenizer(sequence)
    print(model_inputs["input_ids"])

    tokens = tokenizer.tokenize(sequence)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(ids)

    print("-------")
    print(tokenizer.decode(model_inputs["input_ids"]))
    print(tokenizer.decode(ids))

    """
    Model was pretrained.
    Some models don’t add special words   
    """

def run():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    # tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

    tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
    output = model(**tokens)
    print(output)

if __name__ == "__main__":
    # get_model_inputs()
    # special_tokens()
    run()