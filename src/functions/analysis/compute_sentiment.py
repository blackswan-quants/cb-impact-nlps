"""
compute_sentiment.py

This module computes sentiment analysis on speech texts using the FinBERT model.
It provides functions to:
    - Compute sentiment scores for individual texts.
    - Aggregate sentiment confidence scores.
    - Iterate over multiple pickled files to aggregate sentiment data.
    - Process a DataFrame of speeches by computing sentiment using both FinBERT and
      a sentiment analysis pipeline.

The final DataFrame is saved as both a CSV file and a pickled object using a helper module.

Usage:
    python compute_sentiment.py

Dependencies:
    - transformers
    - pandas
    - scipy
    - torch
    - pickle
    - helpermodules (for memory_handling)
"""

import pickle
import pandas as pd
import scipy
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from helpermodules import memory_handling as mh


def finbert_sentiment(text: str, tokenizer, model) -> tuple[float, float, float, str]:
    """
    Compute sentiment scores using FinBERT for a given text.

    Parameters
    ----------
    text : str
        The input text to analyze.
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer corresponding to the FinBERT model.
    model : transformers.PreTrainedModel
        The FinBERT model.

    Returns
    -------
    tuple
        A tuple containing:
            - positive score,
            - negative score,
            - neutral score,
            - the sentiment label with the highest score.
    """
    with torch.no_grad():
        inputs = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        outputs = model(**inputs)
        logits = outputs.logits
        scores = {
            label: score
            for label, score in zip(
                model.config.id2label.values(),
                scipy.special.softmax(logits.numpy().squeeze())
            )
        }
        return (
            scores["positive"],
            scores["negative"],
            scores["neutral"],
            max(scores, key=scores.get)
        )


def analyze_sentiment(text: str, nlp) -> tuple[str, float]:
    """
    Analyze the sentiment of a given text using a sentiment analysis pipeline.

    Parameters
    ----------
    text : str
        Input text (truncated to 512 tokens).
    nlp : transformers.Pipeline
        The sentiment analysis pipeline.

    Returns
    -------
    tuple
        A tuple containing the sentiment label and the confidence score.
    """
    result = nlp(text[:512])
    return result[0]["label"], result[0]["score"]


def aggregate_sentiment_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentiment confidence scores grouped by 'date', 'speaker', and 'sentiment'.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns:
            - 'date': Date of the sentiment entry.
            - 'speaker': Speaker associated with the sentiment.
            - 'sentiment': Sentiment category ('positive', 'neutral', 'negative').
            - 'confidence': Confidence score for the sentiment.

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with summed 'confidence' for each group, sorted by
        'date', 'speaker', and 'sentiment'.
    """
    grouped = (
        df.groupby(['date', 'speaker', 'sentiment'])['confidence']
        .sum()
        .reset_index()
    )
    return grouped.sort_values(by=['date', 'speaker', 'sentiment'])


def aggregate_sentiment_iterator(start_date: int, end_date: int, pickle_file_name: str) -> pd.DataFrame:
    """
    Aggregate sentiment confidence data from multiple pickle files over a specified date range.

    The function iterates over the years from `start_date` to `end_date` (inclusive),
    loading a corresponding pickle file for each year from the "./pickle_files/" directory.
    Each file's name is constructed by appending the year and ".pkl" to the provided
    `pickle_file_name` prefix. The loaded data is aggregated using `aggregate_sentiment_confidence`.

    Parameters
    ----------
    start_date : int
        The starting year (inclusive).
    end_date : int
        The ending year (inclusive).
    pickle_file_name : str
        The base name of the pickle files (without the year and extension).
        Files should be named in the format: "./pickle_files/{pickle_file_name}{year}.pkl".

    Returns
    -------
    pd.DataFrame
        A DataFrame containing aggregated sentiment confidence data from all processed files.
    """
    final_result = pd.DataFrame()
    for year in range(start_date, end_date + 1):
        current_file = f'./pickle_files/{pickle_file_name}{year}.pkl'
        with open(current_file, "rb") as f:
            df_year = pickle.load(f)
        result = aggregate_sentiment_confidence(df_year)
        final_result = pd.concat([final_result, result], ignore_index=True)
    return final_result


def compute_sentiment(df_speech: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sentiment analysis for each row in the speech DataFrame using FinBERT.

    The function performs the following:
      - Drops unnecessary columns ('title', 'link', 'text').
      - Initializes FinBERT's tokenizer and model.
      - Computes FinBERT sentiment scores on 'text_by_minute' for positive, negative,
        neutral scores and the predicted sentiment label.
      - Computes a 'finbert_score' (positive minus negative).
      - Uses a sentiment analysis pipeline to assign a sentiment label and confidence.
      - Saves the final DataFrame as a CSV and a pickle file using a helper module.

    Parameters
    ----------
    df_speech : pd.DataFrame
        DataFrame containing speech data with at least a 'text_by_minute' column.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame with sentiment analysis results.
    """
    df = df_speech.copy()
    # Drop unnecessary columns
    df.drop(["title", "link", "text"], axis=1, inplace=True)

    # Initialize FinBERT tokenizer and model
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Initialize the sentiment analysis pipeline
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    # Compute FinBERT sentiment scores for each row in 'text_by_minute'
    df[["finbert_pos", "finbert_neg", "finbert_neu", "finbert_sentiment"]] = (
        df["text_by_minute"].apply(lambda x: pd.Series(finbert_sentiment(x, tokenizer, model)))
    )
    df["finbert_score"] = df["finbert_pos"] - df["finbert_neg"]

    # Compute pipeline-based sentiment analysis results
    df["sentiment"], df["confidence"] = zip(*df["text_by_minute"].apply(lambda x: analyze_sentiment(x, nlp)))

    # Save the final DataFrame as CSV and pickle
    output_csv = "2020-2024sentiment.csv"
    df.to_csv(output_csv, index=False)
    pickle_helper = mh.PickleHelper(df)
    pickle_helper.pickle_dump('2020-2024sentiment')

    print(df)
    return df


if __name__ == "__main__":
    # Example usage:
    # Load your speech DataFrame from a file (e.g., CSV or pickle) and then compute sentiment.
    # For example:
    # df_speech = pd.read_pickle("path_to_speech_pickle.pkl")
    # compute_sentiment(df_speech)
    pass
