from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import pandas as pd
import scipy
import torch
import pickle
import helper_speech_analysis as hsa


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



def aggregate_sentiment_confidence(df):
    """
    Creates a new DataFrame grouped by 'date' and 'speaker', summing the 'confidence'
    values for each 'sentiment'. The resulting DataFrame contains three rows 
    (one for each sentiment: positive, neutral, negative) for every combination 
    of 'date' and 'speaker'.

    Parameters:
        df (pd.DataFrame): Input DataFrame with the following columns:
                           - 'date': Date of the sentiment entry.
                           - 'speaker': Speaker associated with the sentiment.
                           - 'sentiment': Sentiment category ('positive', 'neutral', 'negative').
                           - 'confidence': Confidence score for the sentiment.

    Returns:
        pd.DataFrame: Aggregated DataFrame with the following columns:
                      - 'date': Unique date for each group.
                      - 'speaker': Unique speaker for each group.
                      - 'sentiment': Sentiment category.
                      - 'confidence': Sum of confidence scores for the sentiment in the group.
    """
    # Group by 'date', 'speaker', and 'sentiment', summing 'confidence'
    grouped = (
        df.groupby(['date', 'speaker', 'sentiment'])['confidence']
        .sum()
        .reset_index()
    )
    
    # Sort for better readability (optional)
    grouped = grouped.sort_values(by=['date', 'speaker', 'sentiment'])
    
    return grouped



def aggregate_sentiment_iterator(start_date, end_date, pickle_file_name):
    """
    Aggregate sentiment confidence data from multiple pickle files over a specified date range.

    This function iterates over the years from `start_date` to `end_date` (inclusive), loading
    a corresponding pickle file for each year from the "./pickle_files/" directory. Each file's name 
    is constructed by appending the year and the ".pkl" extension to the provided `pickle_file_name` prefix.
    The loaded data is then processed using `hsa.aggregate_sentiment_confidence`, and the resulting
    DataFrame is concatenated into a final DataFrame containing aggregated sentiment confidence data 
    across the entire date range.

    Parameters:
        start_date (int): The starting year (inclusive) of the date range.
        end_date (int): The ending year (inclusive) of the date range.
        pickle_file_name (str): The base name of the pickle files (without the year and file extension).
                                Files are expected to be named in the format:
                                "./pickle_files/{pickle_file_name}{year}.pkl".

    Returns:
        pd.DataFrame: A pandas DataFrame containing the aggregated sentiment confidence data from all 
                      processed pickle files.
    """
    
    final_result = pd.DataFrame()

    for i in range(start_date, end_date+1):

        # create the file name string to load
        pickle_file_name_curr = './pickle_files/' + pickle_file_name + str(i) + '.pkl'

        # open the pickle
        df = open(pickle_file_name_curr, "rb")
        df = pickle.load(df)

        # aggregate the data to the current year
        result = hsa.aggregate_sentiment_confidence(df)

        final_result = pd.concat([final_result, result], ignore_index=True)
    
    return final_result
