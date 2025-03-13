"""
This script is part of the 'cb-impact-nlps' project, which analyzes the impact
of Federal Reserve (Fed) speeches on the market using NLP techniques.

FUNCTIONALITY:
1. Loads multiple pickled DataFrames containing Fed speeches, price data, and sentiment.
2. Cleans and standardizes speaker names.
3. Adjusts timestamps to market hours.
4. Filters out speeches outside of market hours.
5. Combines final speech data with price and sentiment data for analysis.
6. Generates plots for the top volatile speech events.

USAGE:
    python main.py

Note: The actual scraping and data retrieval steps (scraping, retrieve_datas, etc.)
are shown as comments to indicate where they *would* be called if needed.
"""

import pandas as pd

# Local imports
from utils import memory_handling as mh
#from functions import compute_sentiment, filtering_df, retrieve_datas, scraping_speeches, update_realtime, analysis
from src.functions.preprocessing.adjust_speech_timestamp import adjust_timestamps as change_time
from src.functions.preprocessing.filtering_df import main as filtering
from src.functions.analysis.analysis_utils import main as plot



def main():
    """
    Main entry point for loading data, cleaning, filtering, and generating plots
    related to Fed speeches and their market impact.
    """

    # -------------------------------------------------------------------------
    # SECTION 1: Load Data
    # -------------------------------------------------------------------------
    yearlist = [2020, 2021, 2022, 2023, 2024]

    # Example: scraping Fed speeches for the years specified.
    # df_fed = scraping(yearlist) -> ALREADY RUN, STORED IN A PICKLE FILE

    speeches_file = "2020-2024fedspeeches.pkl"
    helper = mh.PickleHelper.pickle_load(speeches_file)
    df_fed = helper.obj

    # Clean up speaker column
    df_fed['speaker'] = df_fed['speaker'].str.replace('Speech - ', '', regex=False)
    df_fed['speaker'] = df_fed['speaker'].str.replace('Discussion - ', '', regex=False)

    # Example: retrieve market data
    # df_prices = retrieve_datas(df_speech, deltabefore, deltaafter) -> ALREADY RUN
    prices_file = "2020-2024prices.pkl"
    helper = mh.PickleHelper.pickle_load(prices_file)
    df_prices = helper.obj
    df_prices['date'] = df_prices['datetime'].dt.date

    # Load speech data
    speech_file = "2020-2024speeches.pkl"
    helper = mh.PickleHelper.pickle_load(speech_file)
    df_speech = helper.obj
    df_speech = df_speech[df_speech['date'] >= '2020-01-01']
    df_speech = df_speech.sort_values(['date', 'timestamp'], ascending=True)  # Final shape

    # Load sentiment data
    sentiment_file = "2020-2024sentiment.pkl"
    helper = mh.PickleHelper.pickle_load(sentiment_file)
    df_sentiment = helper.obj

    # -------------------------------------------------------------------------
    # SECTION 2: Adjust Timestamps and Filter Out-of-Market Speeches
    # -------------------------------------------------------------------------
    # Rename columns for clarity
    df_fed.rename(columns={'timestamp': 'opening_time'}, inplace=True)

    # Update the correct timestamp for df_speech using change_time()
    df_speech_final = change_time(df_speech, df_fed)
    df_speech_final = df_speech_final.sort_values(['date', 'timestamp'], ascending=True)

    # Define market open and close times (Eastern Time)
    market_open = pd.to_datetime('09:30:00', format='%H:%M:%S').time()
    market_close = pd.to_datetime('16:00:00', format='%H:%M:%S').time()

    # Extract time from 'timestamp'
    df_speech_final['time'] = df_speech_final['timestamp'].dt.time

    # Calculate how many unique speeches are dropped when filtering outside market hours
    before_filter_count = len(df_speech_final['text'].unique())
    df_filtered = df_speech_final[
        (df_speech_final['time'] >= market_open) & (df_speech_final['time'] < market_close)
    ]
    after_filter_count = len(df_filtered['text'].unique())

    print(
        f"When filtering out data outside of market hours, the drop ratio is "
        f"{(after_filter_count / before_filter_count) * 100:.2f}%. "
        f"We have dropped {before_filter_count - after_filter_count} values."
    )

    # Drop 'time' column after filtering
    df_speech_final = df_filtered.drop(columns=['time'])

    # -------------------------------------------------------------------------
    # SECTION 3: Combine Speech, Price, and Sentiment Data
    # -------------------------------------------------------------------------
    df_speech = df_speech_final  # rename for clarity

    # Example time deltas for retrieving price data before and after speeches
    delta_before = 5
    delta_after = 4

    # Filter and merge dataframes (speeches, prices, sentiment)
    df_speech_final, df_prices_final = filtering(
        df_prices,
        df_speech,
        df_sentiment,
        delta_before,
        delta_after
    )


    # -------------------------------------------------------------------------
    # SECTION 4: Visualization/Analysis
    # -------------------------------------------------------------------------
    # Plot the top_n volatility events around speech time
    plot(df_speech_final, delta_before, delta_after, top_n=3)

    # -------------------------------------------------------------------------
    # NOTES / NEXT STEPS
    # -------------------------------------------------------------------------
    # -> fix the delta function [DONE]
    # -> add volume studies or VWAP bands graph [DONE]
    # -> add function to save word graphs for visualization [DONE]
    # -> correlation analysis (t:t), (t-1:t)
    # -> extend the study back to 2020 [DONE]

    print("Script completed successfully.")

if __name__ == "__main__":
    main()
