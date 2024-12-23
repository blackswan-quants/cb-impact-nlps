import pandas as pd

# Utility Functions
def calculate_speech_durations(df_speech):
    """
    Calculates the duration of each speech by counting the number of rows for each unique
    combination of 'date', 'speaker', and 'speech'.

    Parameters:
    df_speech : pandas.DataFrame
        A dataframe containing ['date', 'timestamp', 'speaker', 'speech'] columns.

    Returns:
    pandas.DataFrame
        A dataframe with a new 'duration' column added, indicating the length of each speech.
    """
    # Ensure the 'date' column is in datetime format
    df_speech['date'] = pd.to_datetime(df_speech['date'])

    # Group by 'date', 'speaker', and 'speech' and count rows to calculate durations
    speech_durations = df_speech.groupby(['date', 'speaker', 'title']).size().reset_index(name='duration')

    # Merge the calculated durations back into the original DataFrame
    df_speech = df_speech.merge(speech_durations, on=['date', 'speaker', 'title'], how='left')

    return df_speech

def find_timestart(df_speech):
    """
    Identifies the earliest timestamp for each speech.

    Parameters:
    df_speech : pandas.DataFrame
        A dataframe containing ['date', 'timestamp', 'speaker', 'speech'] columns.

    Returns:
    pandas.DataFrame
        A dataframe containing only the first timestamp for each unique speech.
    """
    df_speech['timestamp'] = pd.to_datetime(df_speech['timestamp'])

    # Get the row with the minimum 'timestamp' for each group of 'date', 'speech', and 'speaker'
    grouped_df = (
        df_speech.loc[df_speech.groupby(['date', 'title', 'speaker'])['timestamp'].idxmin()]
        .reset_index(drop=True)
    )

    return grouped_df

def filtering(df_prices, df_speech, deltabefore=0, deltaafter=0):
    """
    Filters df_prices by retaining only rows where 'timestamp' falls within the time range
    of speeches in df_speech, including optional buffers before and after the speech duration.

    Parameters:
    df_prices : pandas.DataFrame
        A dataframe containing ['date', 'datetime', 'close', 'volume'] columns.
    df_speech : pandas.DataFrame
        A dataframe containing ['date', 'timestamp', 'speaker', 'speech', 'duration'] columns.
    deltabefore : int, optional
        Time in minutes to include before the start of each speech (default is 0).
    deltaafter : int, optional
        Time in minutes to include after the end of each speech (default is 0).

    Returns:
    pandas.DataFrame
        A filtered dataframe containing rows from df_prices within the speech time ranges.
    """
    # Ensure datetime columns are in the correct format
    df_prices['datetime'] = pd.to_datetime(df_prices['datetime'])

    # Prepare durations and select earliest timestamps
    durations_df = calculate_speech_durations(df_speech)
    durations_df = find_timestart(durations_df)
    durations_df['timestamp'] = durations_df.apply(
        lambda row: row['timestamp'].replace(year=row['date'].year, month=row['date'].month, day=row['date'].day),
        axis=1
    )

    # Initialize list for storing filtered rows
    filtered_rows = []

    # Iterate over each speech and filter df_prices accordingly
    for _, speech in durations_df.iterrows():
        start_time = pd.to_datetime(speech['timestamp']) - pd.Timedelta(minutes=deltabefore)
        duration = speech['duration']
        end_time = pd.to_datetime(speech['timestamp']) + pd.Timedelta(minutes=duration + deltaafter)
        print(start_time,end_time,'\n\n')
        # Filter rows based on time range
        mask = (df_prices['datetime'] >= start_time) & (df_prices['datetime'] <= end_time)
        filtered_subset = df_prices[mask].copy()

        # Add speech-related details
        filtered_subset['title'] = speech['title']
        filtered_subset['speaker'] = speech['speaker']
        filtered_rows.append(filtered_subset)

    # Combine filtered rows into a single dataframe     
    filtered_df = pd.concat(filtered_rows, ignore_index=True)

    return filtered_df

def main(df_prices, df_speech, df_sentiment, deltabefore=0, deltaafter=0):
    """
    Main function to execute the filtering and merging of speech and price data.

    Parameters:
    df_prices : pandas.DataFrame
        A dataframe containing the price data.
    df_speech : pandas.DataFrame
        A dataframe containing speech details.
    df_sentiment : pandas.DataFrame
        A dataframe containing sentiment scores for each speech.
    deltabefore : int, optional
        Time in minutes to include before the start of each speech (default is 0).
    deltaafter : int, optional
        Time in minutes to include after the end of each speech (default is 0).

    Returns:
    pandas.DataFrame, pandas.DataFrame
        The filtered speech and price dataframes.
    """
    
    
    df_speech_final = pd.merge(
        df_speech, 
        df_sentiment[['text_by_minute', 'finbert_score', 'speaker']], 
        on=['text_by_minute', 'speaker'], 
        how='left'
    )
    df_speech_final.rename(columns={'speech': 'title'}, inplace=True)
    

    # Filter prices based on speech data
    df_prices_final = filtering(df_prices, df_speech_final, deltabefore, deltaafter)

    
    # Calculate percentage change in price and merge with speech data
    df_prices_final.rename(columns={'datetime': 'timestamp'}, inplace=True)
    df_prices_final['pct_change'] = df_prices_final.groupby(['title', 'date'])['close'].pct_change()
    df_speech_final.rename(columns={'speech':'title'}, inplace=True)

    df_speech_final['timestamp'] = df_speech_final.apply(
        lambda row: row['timestamp'].replace(year=row['date'].year, month=row['date'].month, day=row['date'].day),
        axis=1
    )

    df_speech_final = pd.merge(
        df_speech_final,
        df_prices_final[['date', 'title', 'timestamp', 'pct_change']],
        on=['date', 'title', 'timestamp'],
        how='left'
    )

    return df_speech_final, df_prices_final

# Execute the script
if __name__ == "__main__":
    df_speech_final, df_prices_final = main()
