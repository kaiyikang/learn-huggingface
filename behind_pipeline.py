""""
(Row Text) -> Tokenizer -> (Input IDs) -> Model -> (Logits) -> Post Processing -> (Predictions)
"""



def tokenize():
    """
    - Splitting the input into words, subwords, or symbols (like punctuation) that are called tokens
    - Mapping each token to an integer
    - Adding additional inputs that may be useful to the model
    """
    from transformers import AutoTokenizer

    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenzier = AutoTokenizer.from_pretrained(checkpoint)

    raw_inputs = [
        "I've been waiting for this session my whole life.",
        "I hate this so much!"
    ]
    inputs = tokenzier(
        raw_inputs,
        padding=True,
        truncation=True,
        return_tensors="pt" # pt=Pytorch, tf=TensorFlow, np=Numpy
        )
    return inputs

def model_hidden_state(inputs):
    """
    AutoModel class which also has a from_pretrained() method.
    It contains only the base Transformer module.
    output:[batch_size, sequence_length, hidden_size]
        - batch_size: The number of sequences processed at a time (2)
        - sequence_length: The length of the numerical representation of the sequence (16)
        - hidden_size: The vector dimension of each model input.(768 or more 3072)
    hidden -> high dimension
    """
    from transformers import AutoModel

    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModel.from_pretrained(checkpoint)

    outputs = model(**inputs)
    print(outputs.last_hidden_state.shape)

def model_classification(inputs):
    """
    AutoModelForSequenceClassification Class with a sequence classification head.
    (to be able to classify the sentences as positive or negative)

    The outputs are the raw, unnormalized scores outputted by the last layer of the model.
    They need to go through a SoftMax layer -> check post_processing() function.
    """
    from transformers import AutoModelForSequenceClassification

    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    outputs = model(**inputs)
    print(outputs.logits.shape)
    print(outputs.logits)
    return outputs

def post_processing(outputs):
    """
    Use model.config.id2label can inspect the labels corresponding to each position.
    """
    import torch

    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    print(predictions)

if __name__ == "__main__":
    # Tokenizer
    inputs = tokenize()

    # Model
    model_hidden_state(inputs=inputs)
    # outputs = model_classification(inputs=inputs)

    # Post Processing
    post_processing(outputs=outputs)