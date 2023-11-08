# """
# Tokenizer: text -> number 
# """
from transformers import BertTokenizer, AutoTokenizer

def based_on_word():
    """
    One word, one number.
    Each word has an ID and it leads to many IDs:
        - dog / dogs
        - run / running
    Need a custom token to represent words that are not in vocabulary, like [UNK].
    """
    print("Kang Yikai is a developer.".split())

def based_on_char():
    """
    Pro: 
    - The vocabulary is much smaller.
    - There are much fewer out-of-vocabulary (unknown) tokens.

    Cons:
    - a very large amount of tokens
    """
    print([c for c in "Kang Yikai is a developer."])

def based_on_subword():
    """
    Principle:
        - frequently used words should not be split into smaller subwords
        - rare words should be decomposed into meaningful subwords
    """
    pass

def encode():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    sequence = "Using a Transformer network is simple! Geschwindigkeitsbegrenzung!"
    tokens = tokenizer.tokenize(sequence)
    print(f"====== Token ====== \n{tokens}")
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"====== IDs ====== \n{ids}")
    decoded_string = tokenizer.decode(ids)
    print(f"====== Decoded string ====== \n{decoded_string}")


if __name__ == "__main__":
    # based_on_word()
    # based_on_char()
    # based_on_subword()
    encode()