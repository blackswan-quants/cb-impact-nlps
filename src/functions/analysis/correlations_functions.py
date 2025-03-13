"""
correlations_functions.py

This module provides functions to analyze the relationship between S&P500 data
and sentiment data. The functionalities include:

    - Calculating the Pearson correlation coefficient.
    - Creating scatter plots to visualize the data.
    - Scaling data using Min-Max scaling.
    - Calculating rolling correlations over a specified window.

Usage:
    python correlations_functions.py

Dependencies:
    - numpy
    - pandas
    - scipy
    - matplotlib
    - scikit-learn
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def calculate_correlation(sp500_data, sentiment_data):
    """
    Calculate the Pearson correlation coefficient between S&P500 data and sentiment.

    Parameters
    ----------
    sp500_data : list or np.array
        Minute-by-minute S&P500 data.
    sentiment_data : list or np.array
        Minute-by-minute sentiment data.

    Returns
    -------
    float
        The Pearson correlation coefficient.
    """
    correlation, _ = pearsonr(sp500_data, sentiment_data)
    return correlation


def plot_scatter(sp500_data, sentiment_data):
    """
    Create a scatter plot to visualize the relationship between S&P500 data and sentiment.

    Parameters
    ----------
    sp500_data : list or np.array
        Minute-by-minute S&P500 data.
    sentiment_data : list or np.array
        Minute-by-minute sentiment data.
    """
    plt.figure(figsize=(10, 6))

    # Scatter plot for S&P500 data
    plt.scatter(range(len(sp500_data)), sp500_data, color='blue', alpha=0.5, label='S&P500')

    # Scatter plot for sentiment data
    plt.scatter(range(len(sentiment_data)), sentiment_data, color='red', alpha=0.5, label='Sentiment')

    plt.title('Scatter Plot of S&P500 and Sentiment Over Time')
    plt.xlabel('Time (Minutes)')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


def scale_data_minmax(sp500_data, sentiment_data):
    """
    Rescale both datasets using Min-Max scaling to a specified range.

    This function rescales the input data to the range [-1, 1] (or any range you define),
    making it easier to compare datasets with different magnitudes while preserving the original distribution.

    Parameters
    ----------
    sp500_data : np.array
        S&P500 data.
    sentiment_data : np.array
        Sentiment data.

    Returns
    -------
    tuple of np.array
        Scaled S&P500 data and scaled sentiment data (both flattened).
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))

    # Reshape data to 2D as required by the scaler
    sp500_data_scaled = scaler.fit_transform(sp500_data.reshape(-1, 1))
    sentiment_data_scaled = scaler.fit_transform(sentiment_data.reshape(-1, 1))

    return sp500_data_scaled.flatten(), sentiment_data_scaled.flatten()


def rolling_correlation(market_data, sentiment, window=5):
    """
    Calculate the rolling correlation between two time series (S&P500 and sentiment).

    !!! BEFORE APPLYING THE FUNCTION, ENSURE THE DATAFRAME IS SORTED BY TIMESTAMP !!!

    Parameters
    ----------
    market_data : pandas.Series
        Series containing S&P500 prices.
    sentiment : pandas.Series
        Series containing minute-by-minute sentiment for a specific speech.
    window : int, optional
        The size of the rolling window for correlation calculation (default is 5).

    Returns
    -------
    pandas.Series
        Rolling correlation values between 'sp500' and 'sentiment'.
    """
    rolling_corr = market_data.rolling(window=window).corr(sentiment)
    return rolling_corr

def calculate_pearson_coefficient(df_speech_final: pd.DataFrame) -> tuple[float, float]:
    """
    Calculate overall Pearson correlation coefficients between price percentage change
    and sentiment score for each unique speech (identified by the 'link' column).

    For each unique speech, the function calculates:
      - The Pearson correlation coefficient between the 'pct_change' and 'finbert_score'.
      - A weighted average of these coefficients, where the weight is the number of data points.
      - Additionally, a second coefficient is calculated using a t+1 shift for the price data.

    Parameters
    ----------
    df_speech_final : pd.DataFrame
        DataFrame containing at least the following columns:
            - 'link': Identifier for each speech.
            - 'pct_change': Percentage change in price.
            - 'finbert_score': Sentiment score from FinBERT.

    Returns
    -------
    tuple of float
        (overall_score_t, overall_score_t1), where:
            - overall_score_t is the weighted average Pearson coefficient at time t.
            - overall_score_t1 is the weighted average Pearson coefficient with a one-minute shift.
    """
    weightlist = []
    pearsonlist = []
    df = df_speech_final.dropna()
    linklist = df.link.unique().tolist()

    # Calculate Pearson coefficient at time t
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

    # Calculate Pearson coefficient with a t+1 shift for price data
    pearsonlist2 = []
    weightlist2 = []
    for link in linklist:
        try:
            sentiment = df[df['link'] == link]['finbert_score']
            indexsentiment = sentiment.index
            shifted_index = [x + 1 for x in indexsentiment]
            prices = df.loc[shifted_index, 'pct_change']
            pearson, _ = pearsonr(prices, sentiment)
            if not np.isnan(pearson):
                weight = len(sentiment)
                pearsonlist2.append(pearson)
                weightlist2.append(weight)
        except Exception as e:
            print(f"Error processing link {link}: {e}")
            continue

    overall_score_t1 = np.average(pearsonlist2, weights=weightlist2)
    return overall_score_t, overall_score_t1



def main():
    """
    Demonstrate the functionality of the correlation analysis module with example data.
    """
    # Example 1: Using random data for scatter plot and correlation calculation
    sp500_data = np.random.randn(100) * 10 + 5000
    sentiment_data = np.random.uniform(-1, 1, 100)  # Replace with actual sentiment data

    print("Correlation (raw data):", calculate_correlation(sp500_data, sentiment_data))
    plot_scatter(sp500_data, sentiment_data)

    # Example 2: Scaling the data and plotting the scaled scatter plot
    sp500_scaled, sentiment_scaled = scale_data_minmax(sp500_data, sentiment_data)
    print("Correlation (scaled data):", calculate_correlation(sp500_scaled, sentiment_scaled))
    plot_scatter(sp500_scaled, sentiment_scaled)

    # Example 3: Rolling correlation on example time series data
    sp500_example = np.array([4000, 4010, 3995, 4005, 4020, 4015])
    sentiment_example = np.array([0.1, 0.2, -0.3, 0.4, -0.5, 0.6])

    # Create timestamps for the example data
    timestamps = pd.date_range('2024-01-01', periods=6, freq='T')

    # Create pandas Series for the example data
    sp500_series = pd.Series(sp500_example, index=timestamps, name='sp500')
    sentiment_series = pd.Series(sentiment_example, index=timestamps, name='sentiment')

    rolling_corr = rolling_correlation(sp500_series, sentiment_series, window=3)
    print("Rolling Correlation (window=3):")
    print(rolling_corr)


if __name__ == "__main__":
    main()
