"""
combine_sentiment.py

This script loads pickled sentiment DataFrames for the years 2020-2024,
concatenates them into a single DataFrame, and then saves the result as both
a CSV file and a pickled object.

Usage:
    python combine_sentiment.py
"""

import os
import sys
import pandas as pd

# Ensure the parent directory is in the system path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import memory_handling as mh


def load_sentiment_data(years):
    """
    Load sentiment data from pickled files for the specified years.

    Parameters
    ----------
    years : list of str
        List of year strings (e.g., ['2020', '2021']).

    Returns
    -------
    pd.DataFrame
        A concatenated DataFrame containing sentiment data for all specified years.
    """
    df_list = []
    for year in years:
        file_name = f"{year}sentiment.pkl"
        helper = mh.PickleHelper.pickle_load(file_name)
        df = helper.obj
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)


def save_sentiment_data(df, csv_filename, pickle_filename):
    """
    Save the sentiment DataFrame to a CSV file and as a pickled object.

    Parameters
    ----------
    df : pd.DataFrame
        The sentiment DataFrame to be saved.
    csv_filename : str
        The filename for the CSV output.
    pickle_filename : str
        The filename (without extension) for the pickle output.
    """
    df.to_csv(csv_filename, index=False)
    pickle_helper = mh.PickleHelper(df)
    pickle_helper.pickle_dump(pickle_filename)


def main():
    """
    Main function to load, combine, and save sentiment data.
    """
    years = ['2020', '2021', '2022', '2023', '2024']
    df_sentiment_final = load_sentiment_data(years)

    # Save as CSV and pickle
    save_sentiment_data(df_sentiment_final, '2020-2024sentiment.csv', '2020-2024sentiment')
    print("Sentiment data combined and saved successfully.")


if __name__ == "__main__":
    main()
