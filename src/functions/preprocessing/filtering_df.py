"""
filtering_df.py

This module provides functions to calculate speech durations, identify the starting timestamp
of each speech, and filter price data based on speech time ranges. The final merging step combines
speech, price, and sentiment data.

Functions:
    - calculate_speech_durations: Computes the duration of each speech.
    - find_timestart: Finds the earliest timestamp for each unique speech.
    - filtering_function: Filters price data to include only rows within speech time ranges,
      including optional buffers.
    - main: Merges speech data with sentiment scores, filters price data, and calculates percentage changes.

Usage:
    The module can be run as a script. When executed, it calls main(), which expects the three dataframes
    (df_prices, df_speech, df_sentiment) as input. Adjust the main() call as needed for your workflow.

Dependencies:
    - pandas
"""

import pandas as pd


def calculate_speech_durations(df_speech: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the duration of each speech by counting rows for each unique combination of 'date',
    'speaker', and 'title'. Only speeches with a duration of 5 minutes or longer are retained.

    Parameters
    ----------
    df_speech : pd.DataFrame
        DataFrame containing columns: ['date', 'timestamp', 'speaker', 'title', ...].

    Returns
    -------
    pd.DataFrame
        DataFrame merged with a new 'duration' column indicating the length of each speech.
    """
    # Ensure 'date' is datetime
    df_speech['date'] = pd.to_datetime(df_speech['date'])
    # Group by date, speaker, and title to calculate duration (number of rows)
    speech_durations = df_speech.groupby(['date', 'speaker', 'title']).size().reset_index(name='duration')
    # Keep only speeches with at least 5 minutes duration
    speech_durations = speech_durations[speech_durations['duration'] >= 5]
    # Merge the durations back into the original dataframe (right merge to keep only long speeches)
    df_speech = df_speech.merge(speech_durations, on=['date', 'speaker', 'title'], how='right')
    return df_speech


def find_timestart(df_speech: pd.DataFrame) -> pd.DataFrame:
    """
    Identify the earliest timestamp for each speech based on 'date', 'title', and 'speaker'.

    Parameters
    ----------
    df_speech : pd.DataFrame
        DataFrame containing columns: ['date', 'timestamp', 'speaker', 'title', ...].

    Returns
    -------
    pd.DataFrame
        DataFrame containing only the first (earliest) timestamp for each unique speech.
    """
    df_speech['timestamp'] = pd.to_datetime(df_speech['timestamp'])
    # Locate the minimum timestamp for each group
    grouped_df = df_speech.loc[df_speech.groupby(['date', 'title', 'speaker'])['timestamp'].idxmin()].reset_index(
        drop=True)
    return grouped_df


def filtering_function(df_prices: pd.DataFrame, df_speech: pd.DataFrame, deltabefore: int = 0,
                       deltaafter: int = 0) -> pd.DataFrame:
    """
    Filter price data (df_prices) by retaining rows where 'datetime' falls within the time range
    of each speech in df_speech, including optional buffers before and after the speech duration.

    Parameters
    ----------
    df_prices : pd.DataFrame
        DataFrame with columns: ['date', 'datetime', 'close', 'volume', ...].
    df_speech : pd.DataFrame
        DataFrame with speech details including ['date', 'timestamp', 'speaker', 'title', 'duration', 'link', ...].
    deltabefore : int, optional
        Number of minutes to include before the start of each speech (default is 0).
    deltaafter : int, optional
        Number of minutes to include after the end of each speech (default is 0).

    Returns
    -------
    pd.DataFrame
        A filtered DataFrame containing rows from df_prices that fall within the specified time ranges.
    """
    # Ensure datetime is properly formatted
    df_prices['datetime'] = pd.to_datetime(df_prices['datetime'])

    # Calculate durations and select earliest timestamps
    initial_count = len(df_speech.link.unique())
    durations_df = calculate_speech_durations(df_speech)
    filtered_count = len(durations_df.link.unique())
    print(f'The number of speeches shorter than 5 minutes is {initial_count - filtered_count}')
    # Select earliest timestamps for each speech
    durations_df = find_timestart(durations_df)
    durations_df['timestamp'] = durations_df.apply(
        lambda row: row['timestamp'].replace(year=row['date'].year, month=row['date'].month, day=row['date'].day),
        axis=1
    )

    filtered_rows = []
    missing_count = 0

    # Iterate over each speech to filter corresponding price data
    for _, speech in durations_df.iterrows():
        start_time = pd.to_datetime(speech['timestamp']) - pd.Timedelta(minutes=deltabefore)
        duration = speech['duration']
        end_time = pd.to_datetime(speech['timestamp']) + pd.Timedelta(minutes=duration + deltaafter)

        mask = (df_prices['datetime'] >= start_time) & (df_prices['datetime'] <= end_time)
        filtered_subset = df_prices[mask].copy()
        if filtered_subset.empty:
            print(f"No price data found for speech: {speech['title']} by {speech['speaker']} on {speech['date']}")
            missing_count += 1
            continue

        # Add speech-related details
        filtered_subset['title'] = speech['title']
        filtered_subset['speaker'] = speech['speaker']
        filtered_rows.append(filtered_subset)

    print(f'{missing_count} speeches have no corresponding price data.')
    filtered_df = pd.concat(filtered_rows, ignore_index=True)
    return filtered_df


def main(df_prices: pd.DataFrame, df_speech: pd.DataFrame, df_sentiment: pd.DataFrame,
         deltabefore: int = 0, deltaafter: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function to merge speech data with sentiment scores, filter price data, and calculate percentage changes.

    The steps include:
        1. Merging speech data with sentiment scores.
        2. Removing duplicate timestamps.
        3. Filtering price data based on speech time ranges.
        4. Calculating percentage changes in price for each speech group.
        5. Merging filtered price data back with speech data.

    Parameters
    ----------
    df_prices : pd.DataFrame
        DataFrame containing price data.
    df_speech : pd.DataFrame
        DataFrame containing speech details.
    df_sentiment : pd.DataFrame
        DataFrame containing sentiment scores.
    deltabefore : int, optional
        Minutes to include before speech start (default 0).
    deltaafter : int, optional
        Minutes to include after speech end (default 0).

    Returns
    -------
    tuple of pd.DataFrame
        (df_speech_final, df_prices_final) after merging and processing.
    """
    # Merge speech data with sentiment scores on common keys
    df_speech_final = pd.merge(
        df_speech,
        df_sentiment[['text_by_minute', 'finbert_score', 'speaker']],
        on=['text_by_minute', 'speaker'],
        how='left'
    )
    df_speech_final.rename(columns={'speech': 'title'}, inplace=True)
    # Eliminate duplicate rows based on timestamp
    df_speech_final = df_speech_final.drop_duplicates(subset='timestamp', keep='first')

    # Filter prices based on speech time ranges
    df_prices_final = filtering_function(df_prices, df_speech_final, deltabefore, deltaafter)

    # Rename datetime column to timestamp and ensure proper formatting
    df_prices_final.rename(columns={'datetime': 'timestamp'}, inplace=True)
    df_prices_final['timestamp'] = pd.to_datetime(df_prices_final['timestamp'])

    def calc_pct_change(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values('timestamp')
        group['pct_change'] = group['close'].pct_change()
        return group

    # Calculate percentage change for each speech group
    df_prices_final = df_prices_final.groupby(['title', 'date'], group_keys=False).apply(calc_pct_change)
    df_speech_final.rename(columns={'speech': 'title'}, inplace=True)

    # Adjust speech timestamps to match the date component
    df_speech_final['timestamp'] = df_speech_final.apply(
        lambda row: row['timestamp'].replace(year=row['date'].year, month=row['date'].month, day=row['date'].day),
        axis=1
    )
    df_speech_final['date'] = pd.to_datetime(df_speech_final['date'])
    df_prices_final['date'] = pd.to_datetime(df_prices_final['date'])

    # Merge speech and price data on date, title, and timestamp
    df_speech_final = pd.merge(
        df_speech_final,
        df_prices_final[['date', 'title', 'timestamp', 'pct_change', 'volume', 'close']],
        on=['date', 'title', 'timestamp'],
        how='right'
    )

    return df_speech_final, df_prices_final




if __name__ == "__main__":
    # Example usage:
    # When running this script, provide df_prices, df_speech, and df_sentiment dataframes.
    # For instance:
    #   df_prices = pd.read_csv("prices.csv")
    #   df_speech = pd.read_csv("speeches.csv")
    #   df_sentiment = pd.read_csv("sentiment.csv")
    # Then call main(df_prices, df_speech, df_sentiment, deltabefore=5, deltaafter=4)
    #
    # Here we just show a placeholder call.
    raise NotImplementedError("Please provide the required dataframes to run main().")
