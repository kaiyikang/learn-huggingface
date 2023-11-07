# """
# Tokenizer: text -> number 
# """
from transformers import BertTokenizer

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

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


if __name__ == "__main__":
    # based_on_word()
    # based_on_char()
    based_on_subword()
