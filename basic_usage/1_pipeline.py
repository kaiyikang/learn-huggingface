from transformers import pipeline
from pprint import pprint as print

"""
Default Download Path:
    /root/.cache/huggingface/transformers
"""

def sentiment_analysis():
    classifier = pipeline('sentiment-analysis')
    return classifier('I love the world')

def zero_shot_classification():
    """
    Specify which labels to use for the classification
    """

    classifier = pipeline('zero-shot-classification')
    return classifier(
        'This is a session about the Transformers library.',
        candidate_labels=['education', 'politics','business']
    )

def text_generation():
    """
    Models and functions are matched according to TAG: 
    https://huggingface.co/models
    """
    generator = pipeline("text-generation", model="distilgpt2")
    return generator(
        'In this session, you will learn how to',
        num_return_sequences=2,
        max_length=15,
    )

def make_filling():
    unmasker = pipeline("fill-mask")
    return unmasker("This session will tell you all about <mask> models", top_k=2)

def named_entity_recognition():
    ner = pipeline("ner", grouped_entities=True)
    return ner("My name is Kang and I work at Valtech_Mobility in Munich.")

if __name__ == "__main__":
    # print(sentiment_analysis())
    # print(zero_shot_classification())
    # print(zero_shot_classification())
    # print(text_generation())
    # print(make_filling())
    print(named_entity_recognition())