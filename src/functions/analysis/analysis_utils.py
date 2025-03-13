"""
analysis_utils.py

This module provides a suite of functions for analyzing the impact of Federal Reserve speeches
on the market using various techniques. Functionalities include:

1. Data normalization and volatility calculations.
2. Extraction of top volatility events.
3. Visualization of sentiment versus cumulative returns.
4. VWAP (Volume Weighted Average Price) calculations and plotting.
5. Word cloud generation for speeches.
6. Density distribution plotting of sentiment scores.
7. Volume vs. sentiment visualization.
8. Calculation of Pearson correlation coefficients (with and without a one-minute shift).
9. Cross-correlation computation between sentiment and price data.
10. Aggregation of average price and sentiment, and merging for combined visualization.

USAGE:
    Run this module as a script for a complete analysis:
        python analysis_utils.py
    or import its functions in your project.

Note: This module assumes the input DataFrame has already been processed to include
columns such as 'date', 'timestamp', 'link', 'pct_change', 'finbert_score', 'speaker', 'title', 'close',
'volume', and 'text'.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from wordcloud import WordCloud
from scipy.stats import pearsonr
from scipy.signal import correlate
from fredapi import Fred
import os
from dotenv import load_dotenv, find_dotenv
from utils import memory_handling as mh
from pathlib import Path


# -------------------------------------------------------------------------
#  Utility Functions
# -------------------------------------------------------------------------
def z_score_standardization(data: np.ndarray) -> np.ndarray:
    """
    Standardizes data using Z-score normalization.

    Parameters
    ----------
    data : np.ndarray
        The data to standardize.

    Returns
    -------
    np.ndarray
        The Z-score standardized data.
    """
    mean_val = np.nanmean(data)
    std_val = np.nanstd(data)
    return (data - mean_val) / (std_val + 1e-10)

# -------------------------------------------------------------------------
#  Volatility Calculation Functions
# -------------------------------------------------------------------------
def volatility_calculator(df_prices_final: pd.DataFrame, deltabefore: int = 0, deltaafter: int = 0) -> pd.DataFrame:
    """
    Calculate daily volatility for the 'close' column within each speech time window.

    Parameters
    ----------
    df_prices_final : pd.DataFrame
        DataFrame containing columns ['date', 'title', 'close', 'pct_change'].
    deltabefore : int, optional
        Number of initial rows to exclude from calculation.
    deltaafter : int, optional
        Number of final rows to exclude from calculation.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['date', 'title', 'volatility'].
    """
    def calculate_volatility(group: pd.DataFrame) -> float:
        if len(group) > deltabefore + deltaafter:
            group_filtered = group.iloc[deltabefore:-deltaafter]
            return group_filtered['pct_change'].std()
        return None

    volatility_series = df_prices_final.groupby(['title', 'date']).apply(calculate_volatility)
    volatility_df = volatility_series.reset_index()
    volatility_df.columns.values[2] = 'volatility'
    volatility_df = volatility_df.dropna()
    final_df = pd.merge(df_prices_final[['date', 'title']].drop_duplicates(), volatility_df, on=['title', 'date'], how='inner')
    return final_df

def get_best_values(volatility_df: pd.DataFrame, number: int) -> pd.DataFrame:
    """
    Select the top records with the highest volatility.

    Parameters
    ----------
    volatility_df : pd.DataFrame
        DataFrame containing ['date', 'volatility', 'title'].
    number : int
        Number of top records to return.

    Returns
    -------
    pd.DataFrame
        DataFrame with the top volatility records.
    """
    volatility_df = volatility_df.dropna(subset=['volatility'])
    top_volatility_df = volatility_df.sort_values(by='volatility', ascending=False).head(number)
    result_df = top_volatility_df[['date', 'volatility', 'title']].drop_duplicates()
    return result_df

def find_best_volatility(top_n: int, df: pd.DataFrame) -> list:
    """
    Identify the top N speeches with the highest volatility (based on pct_change standard deviation).

    Parameters
    ----------
    top_n : int
        Number of top speeches to return.
    df : pd.DataFrame
        DataFrame containing a 'link' column and 'pct_change' values.

    Returns
    -------
    list
        List of links corresponding to the top N speeches.
    """
    df = df.dropna()
    volatility_dict = {link: df[df['link'] == link]['pct_change'].std() for link in df['link'].unique()}
    store_df = pd.DataFrame.from_dict(volatility_dict, orient='index', columns=['volatility']).sort_values(by='volatility', ascending=False)
    return store_df.head(top_n).index.tolist()

def find_longest_speech(top_n: int, df: pd.DataFrame) -> list:
    """
    Identify the top N longest speeches based on the number of rows.

    Parameters
    ----------
    top_n : int
        Number of top speeches to return.
    df : pd.DataFrame
        DataFrame containing a 'link' column.

    Returns
    -------
    list
        List of links corresponding to the longest speeches.
    """
    df = df.dropna()
    length_dict = {link: df[df['link'] == link].shape[0] for link in df['link'].unique()}
    store_df = pd.DataFrame.from_dict(length_dict, orient='index', columns=['length']).sort_values(by='length', ascending=False)
    return store_df.head(top_n).index.tolist()

# -------------------------------------------------------------------------
#  Sentiment vs. Cumulative Return Plotter
# -------------------------------------------------------------------------
def plot_sentiment_vs_cumret(df: pd.DataFrame, df_top_values: pd.DataFrame, deltabefore: int, deltaafter: int, degree: int = 2) -> None:
    """
    Plot sentiment scores and cumulative returns as line plots with a polynomial approximation.

    Parameters
    ----------
    df : pd.DataFrame
        Main DataFrame containing 'pct_change', 'finbert_score', 'timestamp', 'date'.
    df_top_values : pd.DataFrame
        DataFrame with top volatility events containing ['title', 'date'].
    deltabefore : int
        Minutes to highlight before the main segment.
    deltaafter : int
        Minutes to highlight after the main segment.
    degree : int, optional
        Degree of the polynomial fit (default is 2).

    Returns
    -------
    None
    """
    best_title_date_pairs = set(zip(df_top_values['title'], df_top_values['date']))
    df_filtered = df[df.apply(lambda row: (row['title'], row['date']) in best_title_date_pairs, axis=1)]
    for speech_id, group in df_filtered.groupby(['title', 'date']):
        pct_change = group['pct_change'].values * 100
        sentiment_score = group['finbert_score'].values
        times = group['timestamp'].dt.tz_localize(None).values
        display_date = group['date'].dt.strftime('%m/%d/%y').unique()[0]

        sorted_indices = np.argsort(times)
        times = times[sorted_indices]
        pct_change = pct_change[sorted_indices]
        sentiment_score = sentiment_score[sorted_indices]

        pct_change = z_score_standardization(pct_change)
        sentiment_score = z_score_standardization(sentiment_score)

        plt.figure(figsize=(10, 6))
        if deltaafter != 0:
            if deltabefore > 0:
                plt.plot(times[:deltabefore+1], pct_change[:deltabefore+1], color="lightblue", linewidth=0.75)
            plt.plot(times[deltabefore:-deltaafter], pct_change[deltabefore:-deltaafter], color="blue", linewidth=1.5)
            if deltaafter > 0:
                plt.plot(times[-deltaafter-1:], pct_change[-deltaafter-1:], color="lightblue", linewidth=0.75)
        else:
            if deltabefore > 0:
                plt.plot(times[:deltabefore+1], pct_change[:deltabefore+1], color="lightblue", linewidth=0.75)
            plt.plot(times[deltabefore:], pct_change[deltabefore:], color="blue", linewidth=1.5)

        plt.plot(times, sentiment_score, color='red', label='Sentiment Score', linewidth=1.5)
        coeffs = np.polyfit(range(len(sentiment_score)), sentiment_score, degree)
        poly = np.poly1d(coeffs)
        plt.plot(times, poly(range(len(times))), color='blue', linestyle='--', label=f'Poly Approx (deg {degree})')

        plt.xlabel('Time Period', fontsize=12)
        plt.ylabel('Values', fontsize=12)
        plt.title(f'{speech_id} in {display_date}', fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

# -------------------------------------------------------------------------
#  VWAP and Volume vs. Sentiment Plotters
# -------------------------------------------------------------------------
def plot_vwap_by_speech(df: pd.DataFrame, df_top_values: pd.DataFrame, interval: int = 5) -> None:
    """
    Plot price, VWAP, VWAP bands, and sentiment for each unique speech and date.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'timestamp', 'close', 'volume', 'finbert_score', 'title', 'date'.
    df_top_values : pd.DataFrame
        DataFrame with top volatility events (['title', 'date']).
    interval : int, optional
        Interval in minutes for x-axis tick spacing (default is 5).

    Returns
    -------
    None
    """
    best_pairs = set(zip(df_top_values['title'], df_top_values['date']))
    df_filtered = df[df.apply(lambda row: (row['title'], row['date']) in best_pairs, axis=1)]
    for speech, date in df_filtered[['title', 'date']].drop_duplicates().itertuples(index=False):
        group = df[(df['title'] == speech) & (df['date'] == date)]
        time = group['timestamp']
        prices = group['close']
        volumes = group['volume']
        sentiment = group['finbert_score']

        sentiment_valid = sentiment.dropna()
        time_sentiment = time[sentiment_valid.index]

        cumulative_price_volume = 0
        cumulative_volume = 0
        vwap_values = []
        volatility_factors = []
        for i in range(len(prices)):
            volatility_factors.append(prices[:i + 1].std())
        for price, volume in zip(prices, volumes):
            cumulative_price_volume += price * volume
            cumulative_volume += volume
            vwap_values.append(cumulative_price_volume / cumulative_volume if cumulative_volume else None)

        upper_band_1 = [vwap + vol for vwap, vol in zip(vwap_values, volatility_factors)]
        lower_band_1 = [vwap - vol for vwap, vol in zip(vwap_values, volatility_factors)]
        upper_band_2 = [vwap + 1.5 * vol for vwap, vol in zip(vwap_values, volatility_factors)]
        lower_band_2 = [vwap - 1.5 * vol for vwap, vol in zip(vwap_values, volatility_factors)]

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=interval))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax1.plot(time, prices, color='blue', label='Price', linewidth=2)
        ax1.plot(time, vwap_values, color='green', label='VWAP', linewidth=2)
        ax1.fill_between(time, lower_band_1, upper_band_1, color='lightgreen', alpha=0.4, label='VWAP Bands (±1σ)')
        ax1.fill_between(time, lower_band_2, upper_band_2, color='lightblue', alpha=0.3, label='VWAP Bands (±1.5σ)')
        ax2 = ax1.twinx()
        ax2.plot(time_sentiment, sentiment_valid, color='red', label='Sentiment', linewidth=2, linestyle='dashed')
        ax2.set_ylabel('Sentiment', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax1.set_title(f'Price vs VWAP with Bands and Sentiment ({speech} - {date})')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price / VWAP')
        plt.xticks(rotation=45)
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), bbox_transform=ax1.transAxes)
        plt.tight_layout()
        plt.show()

def plot_volumevssentiment(df: pd.DataFrame, linklist: list, interval: int = 1, deltabefore: int = 5, deltaafter: int = 4) -> None:
    """
    Plot volume versus sentiment for selected speeches.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'timestamp', 'close', 'volume', 'finbert_score', 'title', 'date', and 'link'.
    linklist : list
        List of links identifying the speeches to plot.
    interval : int, optional
        Interval for x-axis ticks (default is 1 minute).
    deltabefore : int, optional
        Minutes before the speech to highlight (default is 5).
    deltaafter : int, optional
        Minutes after the speech to highlight (default is 4).

    Returns
    -------
    None
    """
    df_filtered = df[df['link'].isin(linklist)]
    for speech, date in df_filtered[['title', 'date']].drop_duplicates().itertuples(index=False):
        group = df[(df['title'] == speech) & (df['date'] == date)]
        time = group['timestamp']
        volumes = group['volume']
        sentiment = group['finbert_score']
        sentiment_valid = sentiment.dropna()
        time_sentiment = time[sentiment_valid.index]
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=interval))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax1.tick_params(axis='x', rotation=45)
        if deltaafter != 0:
            if deltabefore > 0:
                plt.plot(time[:deltabefore+1], volumes[:deltabefore+1], color="lightblue", linewidth=0.75)
            plt.plot(time[deltabefore:-deltaafter-1], volumes[deltabefore:-deltaafter-1], color="green", label='Volume', linewidth=1.5)
            if deltaafter > 0:
                plt.plot(time[-deltaafter-2:], volumes[-deltaafter-2:], color="lightblue", linewidth=0.75)
        else:
            if deltabefore > 0:
                plt.plot(time[:deltabefore+1], volumes[:deltabefore+1], color="lightblue", linewidth=0.75)
            plt.plot(time[deltabefore:], volumes[deltabefore:], color="green", label='Volume', linewidth=1.5)
        ax2 = ax1.twinx()
        ax2.plot(time_sentiment, sentiment_valid, color='red', label='Sentiment', linewidth=2, linestyle='dashed')
        ax2.set_ylabel('Sentiment', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax1.set_title(f'Volume vs Sentiment ("{speech}" on {date.date()})')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Volume')
        plt.xticks(rotation=45)
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), bbox_transform=ax1.transAxes)
        plt.tight_layout()
        plt.show()

# -------------------------------------------------------------------------
#  Pearson Correlation and Cross-Correlation Functions
# -------------------------------------------------------------------------
def calculate_pearson_coefficient_pct(df_speech_final: pd.DataFrame, linklist: list = None) -> float:
    """
    Calculate a weighted average Pearson coefficient between pct_change and finbert_score.

    Parameters
    ----------
    df_speech_final : pd.DataFrame
        DataFrame containing 'link', 'pct_change', and 'finbert_score'.
    linklist : list, optional
        Specific list of links to calculate the coefficient for; if None, uses all links.

    Returns
    -------
    float
        Weighted average Pearson coefficient (pct_change vs. finbert_score).
    """
    weightlist = []
    pearsonlist = []
    df = df_speech_final.dropna()
    if linklist is None:
        linklist = df.link.unique().tolist()
    for link in linklist:
        try:
            prices = df[df['link'] == link]['pct_change']
            sentiment = df[df['link'] == link]['finbert_score']
            pearson, _ = pearsonr(prices, sentiment)
            if not np.isnan(pearson):
                weight = len(prices)
                pearsonlist.append(pearson)
                weightlist.append(weight)
        except Exception as e:
            print(f"Error processing link {link}: {e}")
            continue
    overall_score_t = np.average(pearsonlist, weights=weightlist)
    return overall_score_t

def calculate_pearson_coefficient(df_speech_final: pd.DataFrame, linklist: list = None) -> float:
    """
    Calculate a weighted average Pearson coefficient between close price and finbert_score.

    Parameters
    ----------
    df_speech_final : pd.DataFrame
        DataFrame containing 'link', 'close', and 'finbert_score'.
    linklist : list, optional
        Specific list of links to calculate the coefficient for; if None, uses all links.

    Returns
    -------
    float
        Weighted average Pearson coefficient (close vs. finbert_score).
    """
    weightlist = []
    pearsonlist = []
    df = df_speech_final.dropna()
    if linklist is None:
        linklist = df.link.unique().tolist()
    for link in linklist:
        try:
            prices = df[df['link'] == link]['close']
            sentiment = df[df['link'] == link]['finbert_score']
            pearson, _ = pearsonr(prices, sentiment)
            if not np.isnan(pearson):
                weight = len(prices)
                pearsonlist.append(pearson)
                weightlist.append(weight)
        except Exception as e:
            print(f"Error processing link {link}: {e}")
            continue
    overall_score_t = np.average(pearsonlist, weights=weightlist)
    return overall_score_t

def compute_cross_correlation_pct(df_speech_final: pd.DataFrame, max_lag: int = 5, linklist: list = None) -> None:
    """
    Compute and plot the cross-correlation between pct_change and finbert_score for each speech.

    Parameters
    ----------
    df_speech_final : pd.DataFrame
        DataFrame containing 'link', 'pct_change', and 'finbert_score'.
    max_lag : int, optional
        Maximum lag (in minutes) to display (default is 5).
    linklist : list, optional
        Specific list of links to process; if None, uses all links.

    Returns
    -------
    None
    """
    df = df_speech_final.dropna()
    if linklist is None:
        linklist = df.link.unique().tolist()

    plt.figure(figsize=(12, 6))
    for link in linklist:
        try:
            prices = df[df['link'] == link]['pct_change'].values
            sentiment = df[df['link'] == link]['finbert_score'].values
            prices = (prices - np.mean(prices)) / np.std(prices)
            sentiment = (sentiment - np.mean(sentiment)) / np.std(sentiment)
            title = df[df['link'] == link]['title'].unique()
            cross_corr = correlate(prices, sentiment, mode="full")
            lags = np.arange(-len(prices) + 1, len(prices))
            mask = (lags >= -max_lag) & (lags <= max_lag)
            lags = lags[mask]
            cross_corr = cross_corr[mask]
            plt.plot(lags, cross_corr, label=f'Title {title}', alpha=0.6)
        except Exception as e:
            print(f"Error processing link {link}: {e}")
            continue
    plt.axvline(0, color='red', linestyle="--", label="Zero Lag")
    plt.xlabel("Lag (minutes)")
    plt.ylabel("Cross-Correlation")
    plt.title("Cross-Correlation between Sentiment and Percent Price Change")
    plt.legend()
    plt.grid()
    plt.show()

def plot_wordscloud(df, df_top_values):
    # Get unique combinations of 'title' and 'date'
    best_speech_dates = df_top_values[['title', 'date']].drop_duplicates()

    # Merge to keep only the rows with the unique 'title' and 'date' pairs
    df = df.merge(best_speech_dates, on=['title', 'date'], how='inner')
    first_rows = df.groupby(['title','date']).first().reset_index()

    for idx, row in first_rows.iterrows():
        if pd.notna(row['text']):
            # Convert speech_text to string explicitly
            speech_text = str(row['text'])
            print(speech_text[:15], '\n\n\n')  # Display the first 15 characters for preview

            # Generate the word cloud for the current speech
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(speech_text)

            # Display the word cloud
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')  # Turn off axis

            # Use the row to access the corresponding values for title and date
            title = row['title']
            date = row['date']
            plt.title(f"Word Cloud for Speech: {title} in {date}")

            plt.show()
        else:
            print("Skipping NaN value")

def compute_cross_correlation(df_speech_final: pd.DataFrame, max_lag: int = 5, linklist: list = None) -> None:
    """
    Compute and plot the cross-correlation between close price and finbert_score for each speech.

    Parameters
    ----------
    df_speech_final : pd.DataFrame
        DataFrame containing 'link', 'close', and 'finbert_score'.
    max_lag : int, optional
        Maximum lag (in minutes) to display (default is 5).
    linklist : list, optional
        Specific list of links to process; if None, uses all links.

    Returns
    -------
    None
    """
    df = df_speech_final.dropna()
    if linklist is None:
        linklist = df.link.unique().tolist()

    plt.figure(figsize=(12, 6))
    for link in linklist:
        try:
            prices = df[df['link'] == link]['close'].values
            sentiment = df[df['link'] == link]['finbert_score'].values
            prices = (prices - np.mean(prices)) / np.std(prices)
            sentiment = (sentiment - np.mean(sentiment)) / np.std(sentiment)
            title = df[df['link'] == link]['title'].unique()
            cross_corr = correlate(prices, sentiment, mode="full")
            lags = np.arange(-len(prices) + 1, len(prices))
            mask = (lags >= -max_lag) & (lags <= max_lag)
            lags = lags[mask]
            cross_corr = cross_corr[mask]
            plt.plot(lags, cross_corr, label=f'Title {title}', alpha=0.6)
        except Exception as e:
            print(f"Error processing link {link}: {e}")
            continue
    plt.axvline(0, color='red', linestyle="--", label="Zero Lag")
    plt.xlabel("Lag (minutes)")
    plt.ylabel("Cross-Correlation")
    plt.title("Cross-Correlation between Sentiment and Close Price Change")
    plt.legend()
    plt.grid()
    plt.show()

# -------------------------------------------------------------------------
#  Price and Sentiment Aggregation and Plotting
# -------------------------------------------------------------------------
def storing_average_price(df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average closing price by date.

    Parameters
    ----------
    df_prices : pd.DataFrame
        DataFrame containing 'date' and 'close'.

    Returns
    -------
    pd.DataFrame
        DataFrame with average 'close' per date, with 'date' as the index.
    """
    average_close = df_prices.groupby('date')['close'].mean().reset_index()
    average_close.set_index('date', inplace=True)
    return average_close

def storing_average_finbert_score(df_speech_final: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average finbert_score by date.

    Parameters
    ----------
    df_speech_final : pd.DataFrame
        DataFrame containing 'date' and 'finbert_score'.

    Returns
    -------
    pd.DataFrame
        DataFrame with average 'finbert_score' per date, with 'date' as the index.
    """
    df = df_speech_final.dropna()
    average_finbert = df.groupby('date')['finbert_score'].mean().reset_index()
    average_finbert.set_index('date', inplace=True)
    return average_finbert

def combine_price_sentiment(prices: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    """
    Merge average price and sentiment data on date.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of average closing prices by date.
    sentiment : pd.DataFrame
        DataFrame of average finbert_score by date.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with 'close' and 'finbert_score' columns.
    """
    combined_df = pd.merge(prices, sentiment, how='left', left_index=True, right_index=True)
    combined_df['finbert_score'].fillna(0, inplace=True)
    return combined_df

def density_distribution (df, info=None):
    df = df.dropna()
    sentiment_score=df.finbert_score
    sns.histplot(sentiment_score, kde=True, bins=20, color='purple', alpha=0.7)
    plt.title(f'Distribution of Sentiment Scores for {info}')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Density')
    plt.show()

def plot_price_with_sentiment(df_combined: pd.DataFrame) -> None:
    """
    Plot closing price and highlight dates where sentiment is non-zero.

    Parameters
    ----------
    df_combined : pd.DataFrame
        DataFrame containing 'close' and 'finbert_score' with date as index.

    Returns
    -------
    None
    """
    plt.plot(df_combined.index, df_combined['close'], color='blue', label='Price')
    sentiment_dates = df_combined[df_combined['finbert_score'] != 0].index
    plt.scatter(sentiment_dates, df_combined.loc[sentiment_dates, 'close'], color='red', label='Sentiment Score > 0')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Price vs Sentiment Score')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------------
#  Helper Function: Fill NaN in 'link'
# -------------------------------------------------------------------------
def fill_link_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN values in the 'link' column by propagating the unique link to rows before and after the first and last occurrence.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'link' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with filled 'link' values.
    """
    unique_links = df['link'].dropna().unique()
    for link in unique_links:
        link_indices = df[df['link'] == link].index
        first_index = link_indices[0]
        last_index = link_indices[-1]
        df.loc[first_index - 5:first_index - 1, 'link'] = link
        df.loc[last_index + 1:last_index + 5, 'link'] = link
    return df


def load_fred_bond_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load US Treasury bond yield data from the FRED API and save it locally as a pickle file.
    If the pickle file already exists, load the data from it instead of making an API call.

    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame
        DataFrame containing bond yield data with a datetime index and a 'Yield' column.
    """
    # Load environment variables from api.env
    dotenv_path = find_dotenv("api.env")
    print("Loading environment from:", dotenv_path)
    load_dotenv(dotenv_path)
    FRED_API_KEY = os.getenv('fred_api')

    project_root = Path(__file__).resolve().parents[3]

    pickle_filename = "2020-2024bond.pkl"
    file_path = project_root / "data" / "pickle_files" / pickle_filename


    # Check if the local pickle file exists
    if os.path.exists(file_path):
        print("Loading bond data from local pickle file...")
        helper = mh.PickleHelper.pickle_load(pickle_filename)
        data = helper.obj
    else:
        print("Local bond pickle file not found. Fetching data from FRED API...")
        fred = Fred(api_key=FRED_API_KEY)
        data = fred.get_series('DGS10', observation_start=start_date, observation_end=end_date)
        helper = mh.PickleHelper(data)
        helper.pickle_dump(pickle_filename)

    # Convert the series to a DataFrame
    df_bond = pd.DataFrame(data, columns=['Yield'])
    df_bond.index.name = 'Date'
    df_bond.index = pd.to_datetime(df_bond.index)
    df_bond['Yield'] = df_bond['Yield'].interpolate(method='linear')
    return df_bond

def extract_valid_event_dates(df_fed: pd.DataFrame, df_bond: pd.DataFrame) -> list:
    """
    Extract the Fed speech event dates that are present in the bond dataset.

    Parameters
    ----------
    df_fed : pd.DataFrame
        DataFrame containing Fed speech data with at least a 'date' column.
    df_bond : pd.DataFrame
        Bond yield DataFrame with a datetime index.

    Returns
    -------
    list
        List of valid event dates (pd.Timestamp) present in the bond data.
    """
    events = dict(zip(df_fed['date'], df_fed['speaker']))
    event_dates = pd.to_datetime(list(events.keys()))
    valid_event_dates = [date for date in event_dates if date in df_bond.index]
    return valid_event_dates


def plot_bond_yield(df_bond: pd.DataFrame, valid_event_dates: list, start_date: str, end_date: str) -> None:
    """
    Plot the US 10-Year Treasury yield over a specific date range and overlay speech event dates.

    Parameters
    ----------
    df_bond : pd.DataFrame
        DataFrame containing bond yield data with a 'Yield' column.
    valid_event_dates : list
        List of event dates (pd.Timestamp) that are valid (present in the bond data).
    start_date : str
        Start date (YYYY-MM-DD) for the plot.
    end_date : str
        End date (YYYY-MM-DD) for the plot.

    Returns
    -------
    None
        Displays a plot.
    """
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # Filter bond data for the specified date range
    df_bond_filtered = df_bond[(df_bond.index >= start_dt) & (df_bond.index <= end_dt)]

    # Filter event dates that fall within the specified range
    event_dates_filtered = [date for date in valid_event_dates if start_dt <= date <= end_dt]
    bond_values_filtered = df_bond.loc[event_dates_filtered, 'Yield']

    plt.figure(figsize=(12, 6))
    plt.plot(df_bond_filtered.index, df_bond_filtered['Yield'], label='10Y Treasury Yield', color='blue')
    plt.scatter(event_dates_filtered, bond_values_filtered, color='red', s=10, zorder=5, label='Speech Dates')

    plt.title(f'US 10-Year Treasury Yield ({start_dt.date()} - {end_dt.date()})')
    plt.xlabel('Date')
    plt.ylabel('Yield (%)')
    plt.legend()
    plt.grid(True)
    plt.show()





# -------------------------------------------------------------------------
#  Main Function
# -------------------------------------------------------------------------
def main(df: pd.DataFrame, deltabefore: int = 0, deltaafter: int = 0, top_n: int = 5, degree: int = 2) -> None:
    """
    Execute a comprehensive analysis by calculating volatility, plotting sentiment vs. cumulative return,
    VWAP, word clouds, density distributions, cross-correlation, and price vs. sentiment trends.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing processed speech, price, and sentiment data.
    deltabefore : int, optional
        Minutes to include before speech start (default is 0).
    deltaafter : int, optional
        Minutes to include after speech end (default is 0).
    top_n : int, optional
        Number of top volatility events to consider (default is 5).
    degree : int, optional
        Degree of polynomial approximation for sentiment plotting (default is 2).

    Returns
    -------
    None
    """
    years = [2020, 2021, 2022, 2023, 2024]
    speakers = {
        'Governor Christopher J. Waller': 'J. Waller',
        'Governor Michelle W. Bowman': 'Michelle Bowman',
        'Vice Chair for Supervision Randal K. Quarles': 'Randal Quarles'
    }



    # Calculate volatility and get top volatility events
    volatility = volatility_calculator(df, deltabefore=deltabefore, deltaafter=deltaafter)
    best_volatility = get_best_values(volatility, number=top_n)

    # Plot sentiment vs. cumulative return
    plot_sentiment_vs_cumret(df, best_volatility, deltabefore, deltaafter, degree=degree)

    # Plot VWAP bands with volume
    plot_vwap_by_speech(df, best_volatility)

    # Plot word cloud for top events
    plot_wordscloud(df, best_volatility)

    # Plot density distribution for selected speakers
    for speaker, label in speakers.items():
        df_speaker = df[df['speaker'] == speaker]
        density_distribution(df_speaker, label)

    # Plot density distribution by year
    dfs_by_year = {year: df[(df['date'] >= f'{year}-01-01') & (df['date'] <= f'{year}-12-31')] for year in years}
    density_distribution(df, 'overall')
    for year, df_year in dfs_by_year.items():
        density_distribution(df_year, str(year))

    # Cross-correlation analysis: top volatility events
    linklistvol = find_best_volatility(3, df)
    # Cross-correlation analysis: longest speeches
    linklistlong = find_longest_speech(3, df)

    # Fill missing 'link' values
    df_nan_filled = fill_link_nan(df)

    # Plot volume vs. sentiment for selected speeches
    plot_volumevssentiment(df_nan_filled, linklistvol)
    plot_volumevssentiment(df_nan_filled, linklistlong)

    # Compute cross-correlation (pct_change vs sentiment)
    compute_cross_correlation_pct(df, linklist=linklistvol)
    # Compute cross-correlation (close vs sentiment) for top volatility events
    compute_cross_correlation(df, linklist=linklistvol)
    # Compute cross-correlation for longest speeches
    compute_cross_correlation_pct(df, linklist=linklistlong)
    compute_cross_correlation(df, linklist=linklistlong)

    # Aggregate price and sentiment by date and plot
    prices = storing_average_price(df)
    sentiment = storing_average_finbert_score(df)
    df_combined = combine_price_sentiment(prices, sentiment)
    plot_price_with_sentiment(df_combined)

    # Define the CSV directory and load Fed data
    csv_dir = os.path.join("..", "data", "csv_files")
    df_fed = pd.read_csv(os.path.join(csv_dir, "2020-2024fed.csv"))

    # Define the overall date range
    start_date = '2020-01-01'
    end_date = '2024-12-31'

    # Load bond data from FRED (or local pickle)
    df_bond = load_fred_bond_data(start_date, end_date)

    # Extract valid event dates from Fed data that exist in the bond dataset
    valid_event_dates = extract_valid_event_dates(df_fed, df_bond)

    # Plot bond yield for each year between 2020 and 2024
    for year in range(2020, 2025):
        plot_bond_yield(df_bond, valid_event_dates, f"{year}-01-01", f"{year}-12-31")

if __name__ == "__main__":
    # Example usage: load a DataFrame from a file (CSV or pickle) and call main(df)
    # For demonstration, this is left as a placeholder.
    pass
