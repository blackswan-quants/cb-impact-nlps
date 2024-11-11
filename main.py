import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import pandas as pd
import scipy
import torch
import helper_speech_analysis as sa

df = open ("./pickle_files/fedspeeches_preprocessed.pkl", "rb")
df = pickle.load(df)
print(df)

tokenizer, model = sa.load_finbert()
pl = sa.build_pipeline(tokenizer, model)

# Notice that this is the raw text, no preprocessing
df[["finbert_pos", "finbert_neg", "finbert_neu", "finbert_sentiment"]] = (
    df["text_by_minute"].apply(sa.finbert_sentiment(tokenizer, model)).apply(pd.Series)
)
df["finbert_score"] = df["finbert_pos"] - df["finbert_neg"]


# Applichiamo l'analisi del sentiment a ogni riga
df["sentiment"], df["confidence"] = zip(*df["text"].apply(sa.analyze_sentiment(pl)))

# Salviamo i risultati in un nuovo file CSV
df.to_csv("transcript_sentiment_analysis.csv", index=False)

