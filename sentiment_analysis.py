import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import pandas as pd
import scipy
import torch
import helper_speech_analysis as sa
from helpermodules import memory_handling as mh


df = open ("./pickle_files/fedspeeches_preprocessed.pkl", "rb")
df = pickle.load(df)
df.drop(["title","link","text"], axis = 1, inplace=True)
print(df)

model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

df = df[df["date"].dt.year == 2020]
#df = df.head()
print(df)


# Notice that this is the raw text, no preprocessing
df[["finbert_pos", "finbert_neg", "finbert_neu", "finbert_sentiment"]] = (
    df["text_by_minute"].apply(lambda x: pd.Series(sa.finbert_sentiment(x, tokenizer, model)))
)
df["finbert_score"] = df["finbert_pos"] - df["finbert_neg"]


# Applichiamo l'analisi del sentiment a ogni riga
df["sentiment"], df["confidence"] = zip(*df["text_by_minute"].apply(lambda x : sa.analyze_sentiment(x, nlp)))

# Salviamo i risultati in un nuovo file CSV
df.to_csv("sentiment_2020.csv", index=False)
pickle_helper = mh.PickleHelper(df)
pickle_helper.pickle_dump('fedspeechees_sentiment_2020')

print(df)



