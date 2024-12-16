import pandas as pd
import numpy as np


def change_time(df_speech, dfnew):
    """
    Adjusts the timestamps in the `df_speech` DataFrame based on the opening time
    from `dfnew` for each speech. Specifically, it updates the timestamp of the
    first event for each unique combination of speaker, date, and title to the
    opening time specified in `dfnew`, and shifts subsequent timestamps by one minute.

    Parameters:
    df_speech (pd.DataFrame): DataFrame containing speech details with columns like
                               'speaker', 'date', 'title', and 'timestamp'.
    dfnew (pd.DataFrame): DataFrame containing speech details with columns 'speaker',
                          'date', 'title', and 'opening_time' (the time when the speech begins).

    Returns:
    pd.DataFrame: A modified version of `df_speech` where the timestamps are updated
                  based on the opening time for matching speeches, and rows with no matches
                  are removed. The rows where the "timestamp" was updated will remain.

    Side Effects:
    - Adds a "check" column to track which rows have been updated.
    - Drops rows that didn't have a matching speech in `dfnew`.
    - Prints the drop ratio, which is the ratio of remaining unique texts after the update.
    """  


    #ensuring the time columns share the same datetime format 
    df_speech.loc[:, 'date'] = pd.to_datetime(df_speech['date'])
    dfnew['date'] = pd.to_datetime(dfnew['date'])

    # Add a "check" column to track updated rows
    df_speech.loc[:, 'check'] = 0

    # Count the initial number of unique texts
    r2 = df_speech['text'].nunique()

    for i in range(len(dfnew)):
        # Extract details for the current speech
        speaker = dfnew.iloc[i]['speaker']
        date = dfnew.iloc[i]['date']
        title = dfnew.iloc[i]['title']
        newtime = dfnew.iloc[i]['opening_time']

        # Find rows in df_speech matching speaker, date, and title
        mask = (df_speech['speaker'] == speaker) & (df_speech['date'] == date) & (df_speech['title'] == title)

        # Update the minimum timestamp and set the "check" flag
        if mask.any():
            min_idx = df_speech.loc[mask, 'timestamp'].idxmin()
            df_speech.at[min_idx, 'timestamp'] = newtime
            df_speech.loc[mask, 'timestamp'] = (
                newtime + pd.to_timedelta(range(len(df_speech[mask])), unit='min')
            )
            df_speech.loc[mask, 'check'] = 1

    # Drop rows that were not updated (check == 0)
    df_speech = df_speech[df_speech['check'] == 1].drop(columns=['check'])

    # Count the remaining unique texts and calculate the drop ratio
    r1 = df_speech['text'].nunique()
    print('The drop ratio is', (r1 / r2)*100,'%')

    return df_speech