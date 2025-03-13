"""
split_text_by_minute.py

This module defines the SpeechProcessor class, which reads speech data from a CSV file,
calculates speech length based on an average speaking rate, splits speech text into 
minute-based chunks, expands the DataFrame accordingly, and saves the preprocessed data.

Usage:
    processor = SpeechProcessor('fedspeeches.csv', timezone='US/Pacific', words_per_minute=150)
    processor.process_speeches()
    processor.save_preprocessed_data('fedspeeches_preprocessed')

Dependencies:
    - pandas
    - pytz
    - utils.memory_handling (local module)
"""

import pandas as pd
import pytz
from utils import memory_handling as mh


class SpeechProcessor:
    def __init__(self, csv_file: str, timezone: str = 'US/Eastern', words_per_minute: int = 130):
        """
        Initialize the SpeechProcessor with a CSV file, timezone, and average speaking rate.

        Parameters
        ----------
        csv_file : str
            Path to the CSV file containing speech data.
        timezone : str, optional
            Timezone for the speech timestamps (default is 'US/Eastern').
        words_per_minute : int, optional
            Average speaking rate in words per minute (default is 130).
        """
        self.df = pd.read_csv(csv_file)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.est = pytz.timezone(timezone)
        self.words_per_minute = words_per_minute

        # Set a default speech start time (10:00 AM localized to the specified timezone)
        self.df['timestamp'] = self.df['date'].apply(
            lambda x: self.est.localize(x.replace(hour=10, minute=0, second=0))
        )

        # Estimate speech length in minutes (at least 1 minute)
        self.df['speech_length_minutes'] = self.df['text'].apply(
            lambda x: max(1, len(x.split()) / self.words_per_minute)
        )

        self.df_expanded = None  # This will hold the expanded DataFrame after processing

    @staticmethod
    def split_text_by_minute(text: str, minutes: int) -> list[str]:
        """
        Split the given text into chunks representing one minute of speech each.

        Parameters
        ----------
        text : str
            The full speech text to be split.
        minutes : int
            The number of minutes the speech lasts.

        Returns
        -------
        list[str]
            A list of text chunks, each corresponding to one minute of speech.
        """
        words = text.split()
        words_per_minute = max(1, len(words) // minutes)
        return [' '.join(words[i:i + words_per_minute]) for i in range(0, len(words), words_per_minute)]

    def process_speeches(self) -> None:
        """
        Process the speeches by splitting the text into minute-based chunks and expanding the DataFrame.

        This method:
          - Applies the split_text_by_minute function to create a 'text_by_minute' column.
          - Explodes the DataFrame so each row represents one minute of speech.
          - Adjusts the 'timestamp' column by adding minute offsets.
          - Drops temporary columns.
        """
        self.df['text_by_minute'] = self.df.apply(
            lambda row: self.split_text_by_minute(row['text'], int(row['speech_length_minutes'])), axis=1
        )
        df_expanded = self.df.explode('text_by_minute').reset_index(drop=True)
        df_expanded['minute'] = df_expanded.groupby('timestamp').cumcount()
        df_expanded['timestamp'] = df_expanded['timestamp'] + pd.to_timedelta(df_expanded['minute'], unit='m')
        df_expanded = df_expanded.drop(columns=['minute', 'speech_length_minutes'])
        self.df_expanded = df_expanded

    def save_preprocessed_data(self, filename: str) -> None:
        """
        Save the preprocessed DataFrame using a custom PickleHelper class.

        Parameters
        ----------
        filename : str
            The filename (without extension) to save the preprocessed data.
        """
        pickle_helper = mh.PickleHelper(self.df_expanded)
        pickle_helper.pickle_dump(filename)

# Example usage:
# processor = SpeechProcessor('fedspeeches.csv', timezone='US/Pacific', words_per_minute=150)
# processor.process_speeches()
# processor.save_preprocessed_data('fedspeeches_preprocessed')
