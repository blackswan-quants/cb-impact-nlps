import pandas as pd
import pytz
import re
from libs.helpermodules import memory_handling as mh

# Load the data
df = pd.read_csv('fedspeeches.csv')

# Convert the date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Define the timezone for EST
est = pytz.timezone('US/Eastern')

# Add a timestamp column (assuming speeches start at 10:00 AM EST)
df['timestamp'] = df['date'].apply(lambda x: est.localize(x.replace(hour=10, minute=0, second=0)))

# Estimate the length of the speech in minutes (assuming average speaking rate of 130 words per minute)
df['speech_length_minutes'] = df['text'].apply(lambda x: max(1, len(x.split()) / 130))


def split_text_by_minute(text, minutes):
    words = text.split()  # Split by whitespace
    words_per_minute = max(1, len(words) // minutes)
    return [' '.join(words[i:i + words_per_minute]) for i in range(0, len(words), words_per_minute)]


# Apply the function to split the text
df['text_by_minute'] = df.apply(lambda row: split_text_by_minute(row['text'], int(row['speech_length_minutes'])),
                                axis=1)

# Expand the dataframe
df_expanded = df.explode('text_by_minute').reset_index(drop=True)

# Add minute offset to the timestamp
df_expanded['minute'] = df_expanded.groupby('timestamp').cumcount()
df_expanded['timestamp'] = df_expanded['timestamp'] + pd.to_timedelta(df_expanded['minute'], unit='m')

# Drop unnecessary columns
df_expanded = df_expanded.drop(columns=['minute', 'speech_length_minutes'])

pickle_helper = mh.PickleHelper(df_expanded)
pickle_helper.pickle_dump('fedspeeches_preprocessed')
