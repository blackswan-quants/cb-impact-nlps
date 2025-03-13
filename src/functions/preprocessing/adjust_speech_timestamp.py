"""
adjust_speech_timestamps.py

This module adjusts the timestamps in a speech DataFrame based on the opening time provided
in another DataFrame. For each matching speech (by speaker, date, and title), it resets the
first timestamp to the opening time and shifts subsequent timestamps by one-minute increments.

The module performs the following:
    - Ensures datetime columns in both DataFrames share the same format.
    - Updates the timestamp of the earliest row in each matching group to the new opening time.
    - Shifts subsequent timestamps by one minute per row.
    - Drops rows that did not match any entry in the opening time DataFrame.
    - Normalizes the time components to match the corresponding date and localizes the timestamps.

Usage:
    adjusted_df = adjust_speech_timestamps(df_speech, df_opening)

Dependencies:
    - pandas
"""

import pandas as pd


def adjust_timestamps(df_speech: pd.DataFrame, df_opening: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust the timestamps in the df_speech DataFrame based on the opening times in df_opening.

    For each speech (matching on date, title, and/or speaker), this function:
      - Updates the earliest timestamp to the specified opening time.
      - Shifts subsequent timestamps by one-minute increments.
      - Marks updated rows with a temporary "check" column and drops rows that were not updated.
      - Normalizes the timestamps so that their date matches the 'date' column and localizes them
        to 'America/New_York'. It also applies an offset adjustment for Daylight Saving Time.

    Parameters
    ----------
    df_speech : pd.DataFrame
        DataFrame containing speech details with columns including 'speaker', 'date', 'title',
        'timestamp', and 'text'.
    df_opening : pd.DataFrame
        DataFrame containing opening time information with columns 'speaker', 'date', 'title',
        and 'opening_time'.

    Returns
    -------
    pd.DataFrame
        Modified df_speech DataFrame with updated timestamps based on the opening times.
    """
    # Ensure date columns are datetime
    df_speech['date'] = pd.to_datetime(df_speech['date'])
    df_opening['date'] = pd.to_datetime(df_opening['date'])

    # Add a temporary column to track updated rows
    df_speech['check'] = 0

    # Count initial unique texts
    initial_unique = df_speech['text'].nunique()

    # Iterate over each row in df_opening to update timestamps in df_speech
    for _, row in df_opening.iterrows():
        speaker = row['speaker']
        date_val = row['date']
        title = row['title']
        new_time = pd.to_datetime(row['opening_time'])  # ensure new_time is a Timestamp

        # Create a mask for matching rows
        mask = (
                (df_speech['date'] == date_val) &
                ((df_speech['title'] == title) | (df_speech['speaker'] == speaker))
        )

        if mask.any():
            # Find the row with the minimum timestamp within the group
            min_idx = df_speech.loc[mask, 'timestamp'].idxmin()
            # Update the first timestamp and shift subsequent ones
            df_speech.at[min_idx, 'timestamp'] = new_time
            num_rows = df_speech.loc[mask].shape[0]
            df_speech.loc[mask, 'timestamp'] = new_time + pd.to_timedelta(range(num_rows), unit='min')
            df_speech.loc[mask, 'check'] = 1

    # Remove rows that did not get updated
    df_speech = df_speech[df_speech['check'] == 1].drop(columns=['check'])

    # Report drop ratio
    remaining_unique = df_speech['text'].nunique()
    drop_ratio = (1 - (remaining_unique / initial_unique)) * 100
    print(f"Drop ratio: {drop_ratio:.2f}% ({initial_unique - remaining_unique} values dropped)")

    # Adjust timestamps: Replace hour, minute, second of date with those from timestamp
    df_speech['timestamp'] = df_speech.apply(
        lambda row: row['date'].replace(
            hour=row['timestamp'].hour, minute=row['timestamp'].minute, second=row['timestamp'].second
        ),
        axis=1
    )

    # Localize timestamps to US Eastern Time
    df_speech['timestamp'] = df_speech['timestamp'].dt.tz_localize('America/New_York')

    # Correct for potential DST offset: if offset is -4 hours, add one hour
    df_speech.loc[
        df_speech['timestamp'].apply(lambda x: x.tzinfo.utcoffset(x) == pd.Timedelta(hours=-4)),
        'timestamp'
    ] += pd.Timedelta(hours=1)

    return df_speech

# Example usage:
# Assuming df_speech and df_opening are pre-loaded DataFrames:
# adjusted_df = adjust_timestamps(df_speech, df_opening)
# print(adjusted_df.head())
