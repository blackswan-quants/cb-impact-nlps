"""
datapreprocessing.py

This module preprocesses the Federal Reserve speeches data by performing the following steps:
    1. Loads raw speech data from a CSV file.
    2. Converts date columns and localizes timestamps to US Eastern Time.
    3. Estimates the speech length in minutes based on word count.
    4. Splits the speech text into minute-by-minute segments.
    5. Expands the DataFrame so that each row represents one minute of speech.
    6. Adjusts timestamps to reflect the minute offsets.
    7. Saves the preprocessed DataFrame as a pickle file.

Usage:
    python datapreprocessing.py

Dependencies:
    - pandas
    - pytz
    - re         (standard library, no installation required)
    - utils.memory_handling (local module for pickle operations)
"""

import pandas as pd
import pytz
import re
from utils import memory_handling as mh


def load_raw_data(csv_filepath: str) -> pd.DataFrame:
    """
    Load the raw Fed speeches data from a CSV file.

    Parameters
    ----------
    csv_filepath : str
        The file path to the CSV file containing the raw speeches.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the raw speech data.
    """
    df = pd.read_csv(csv_filepath)
    df['date'] = pd.to_datetime(df['date'])
    return df


def add_timestamp_and_length(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a timestamp column (defaulting to 10:00 AM EST) and estimate the speech length in minutes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the raw speech data.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with 'timestamp' and 'speech_length_minutes' columns.
    """
    # Define the US Eastern timezone
    est = pytz.timezone('US/Eastern')

    # Add a timestamp column assuming speeches start at 10:00 AM EST
    df['timestamp'] = df['date'].apply(lambda x: est.localize(x.replace(hour=10, minute=0, second=0)))

    # Estimate speech length in minutes (using an average speaking rate of 130 words per minute)
    df['speech_length_minutes'] = df['text'].apply(lambda x: max(1, len(x.split()) / 130))

    return df


def split_text_by_minute(text: str, minutes: int) -> list:
    """
    Split the given text into segments corresponding to each minute of speech.

    Parameters
    ----------
    text : str
        The full text of the speech.
    minutes : int
        Estimated length of the speech in minutes.

    Returns
    -------
    list
        A list of strings, each representing the text for one minute of the speech.
    """
    words = text.split()
    words_per_minute = max(1, len(words) // minutes)
    return [' '.join(words[i:i + words_per_minute]) for i in range(0, len(words), words_per_minute)]


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the raw speech DataFrame by splitting texts into minute segments,
    expanding the DataFrame, and adjusting timestamps.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing raw speech data with 'text', 'timestamp', and 'speech_length_minutes'.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame where each row corresponds to one minute of speech.
    """
    # Split the text into minute-by-minute segments
    df['text_by_minute'] = df.apply(
        lambda row: split_text_by_minute(row['text'], int(row['speech_length_minutes'])),
        axis=1
    )

    # Expand the DataFrame so that each row represents one minute of speech
    df_expanded = df.explode('text_by_minute').reset_index(drop=True)

    # Add minute offset based on the group of rows with the same timestamp
    df_expanded['minute'] = df_expanded.groupby('timestamp').cumcount()

    # Adjust timestamp by adding the minute offset
    df_expanded['timestamp'] = df_expanded['timestamp'] + pd.to_timedelta(df_expanded['minute'], unit='m')

    # Drop unnecessary columns
    df_expanded = df_expanded.drop(columns=['minute', 'speech_length_minutes'])

    return df_expanded


def save_preprocessed_data(df: pd.DataFrame, pickle_filename: str):
    """
    Save the preprocessed DataFrame as a pickle file using the memory_handling module.

    Parameters
    ----------
    df : pd.DataFrame
        The preprocessed DataFrame.
    pickle_filename : str
        The filename (without extension) for the pickle dump.
    """
    pickle_helper = mh.PickleHelper(df)
    pickle_helper.pickle_dump(pickle_filename)


def main():
    """
    Main entry point for preprocessing the Fed speeches data.
    """
    # Adjust the CSV filepath as necessary
    csv_filepath = '../csv_files/fedspeeches.csv'

    # Load raw data
    df_raw = load_raw_data(csv_filepath)

    # Add timestamp and calculate estimated speech length
    df_processed = add_timestamp_and_length(df_raw)

    # Preprocess data by splitting texts and adjusting timestamps
    df_preprocessed = preprocess_data(df_processed)

    # Save the preprocessed DataFrame as a pickle file
    save_preprocessed_data(df_preprocessed, 'fedspeeches_preprocessed')

    print("Data preprocessing completed and saved as 'fedspeeches_preprocessed'")


if __name__ == "__main__":
    main()
