from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import pandas as pd
import scipy
import torch


def finbert_sentiment(text: str, tokenizer, model) -> tuple[float, float, float, str]:
    

    with torch.no_grad():
        inputs = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        outputs = model(**inputs)
        logits = outputs.logits
        scores = {
            k: v
            for k, v in zip(
                model.config.id2label.values(),
                scipy.special.softmax(logits.numpy().squeeze()),
            )
        }
        return (
            scores["positive"],
            scores["negative"],
            scores["neutral"],
            max(scores, key=scores.get),
        )
    
# Funzione per analizzare il sentiment di ogni riga
def analyze_sentiment(text, nlp):
    result = nlp(text[:512])  # FinBERT ha un limite di 512 token
    return result[0]["label"], result[0]["score"]