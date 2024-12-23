# takes as argument: number of top values, deltabefore, deltaafter 



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Utility Function
def z_score_standardization(data):
    """
    Standardizes data using Z-score normalization.

    Parameters:
    data : array-like
        The data to standardize.

    Returns:
    array-like
        The Z-score standardized data.
    """
    mean_val = np.nanmean(data)  # Ignore NaNs when calculating mean
    std_val = np.nanstd(data)    # Ignore NaNs when calculating std
    return (data - mean_val) / (std_val + 1e-10)


# Volatility Calculation Function
def volatility_calculator(df_prices_final, deltabefore=0, deltaafter=0):
    """
    Calculate daily volatility for the 'close' column in the dataframe.

    Parameters:
    df_prices_final : pandas.DataFrame
        A dataframe containing ['date', 'speech', 'close', 'volume'] columns.
    deltabefore : int, optional
        Number of initial rows to exclude from the calculation.
    deltaafter : int, optional
        Number of final rows to exclude from the calculation.

    Returns:
    pandas.DataFrame
        A dataframe containing ['date', 'title', 'volatility'] columns.
    """
    
    # Filter rows based on deltabefore and deltaafter
    if deltaafter != 0:
        filtered_df = df_prices_final.iloc[deltabefore:-deltaafter]
        volatility_series = filtered_df.groupby('title')['pct_change'].std()
    else:
        filtered_df = df_prices_final.iloc[deltabefore:]
        volatility_series = filtered_df.groupby('title')['pct_change'].std()
    
    # Reset index and rename column
    volatility_df = volatility_series.reset_index()
    volatility_df.rename(columns={'pct_change': 'volatility'}, inplace=True)

    # Merge with original dataframe to include 'date' and 'speech'
    final_df = pd.merge(df_prices_final[['date', 'title']].drop_duplicates(), 
                        volatility_df, 
                        on='title', 
                        how='inner')
    
    return final_df


# Get Top Volatility Values Function
def get_best_values(volatility_df, number):
    """
    Get the top dates with the highest volatility from the given dataframe.

    Parameters:
    volatility_df : pandas.DataFrame
        A dataframe containing ['date', 'speech', 'volatility'] columns.
    number : int
        The number of top volatility records to return.

    Returns:
    pandas.DataFrame
        A dataframe with the top 'number' records, containing ['date', 'volatility'] columns.
    """
    # Drop NaN values and sort by volatility in descending order
    volatility_df = volatility_df.dropna(subset=['volatility'])
    top_volatility_df = volatility_df.sort_values(by='volatility', ascending=False).head(number)

    # Select the relevant columns
    result_df = top_volatility_df[['date', 'volatility', 'title']].drop_duplicates()

    return result_df


# Sentiment vs Cumulative Return Plotter Function
def plot_sentiment_vs_cumret(df, df_top_values, deltabefore, deltaafter, degree=2):
    """
    Plots sentiment scores and cumulative returns as line plots, with a polynomial approximation.

    Parameters:
    df : pandas.DataFrame
        The main dataframe containing data to plot.
    df_top_values : pandas.DataFrame
        Dataframe containing the top values to filter and plot.
    deltabefore : int
        Number of initial points to highlight in lighter color.
    deltaafter : int
        Number of final points to highlight in lighter color.
    degree : int, optional
        Degree of the polynomial approximation (default is 2).

    Returns:
    None
    """
    # Filter for top values based on speech
    best_speech = df_top_values['title'].unique().tolist()
    df_filtered = df[df['title'].isin(best_speech)]

    for speech_id, speech_group in df_filtered.groupby('title'):
        pct_change = speech_group['pct_change'].values * 100
        sentiment_score = speech_group['finbert_score'].values
        time = speech_group['timestamp'].dt.tz_localize(None).values  # Remove timezone info
        date = speech_group['date'].dt.strftime('%m/%d/%y').unique()[0]

        # Ensure time is sorted
        sorted_indices = np.argsort(time)
        time = time[sorted_indices]
        pct_change = pct_change[sorted_indices]
        sentiment_score = sentiment_score[sorted_indices]

        # Apply z-score standardization to pct_change and sentiment_score
        pct_change = z_score_standardization(pct_change)
        sentiment_score = z_score_standardization(sentiment_score)

        # Plot pct_change and sentiment_score
        plt.figure(figsize=(10, 6))

        # Plot pct_change
        if deltaafter != 0:
            if deltabefore > 0:
                plt.plot(time[:deltabefore], pct_change[:deltabefore], color="lightblue", linewidth=0.75)
            plt.plot(time[deltabefore:-deltaafter], pct_change[deltabefore:-deltaafter], color="blue", linewidth=1.5)
            if deltaafter > 0:
                plt.plot(time[-deltaafter:], pct_change[-deltaafter:], color="lightblue", linewidth=0.75)
        else:
            if deltabefore > 0:
                plt.plot(time[:deltabefore], pct_change[:deltabefore], color="lightblue", linewidth=0.75)
            plt.plot(time[deltabefore:], pct_change[deltabefore:], color="blue", linewidth=1.5)

        # Plot sentiment
        plt.plot(time, sentiment_score, color='red', label='Sentiment Score', linewidth=1.5)

        # Polynomial fit for sentiment_score
        coeffs_sentiment = np.polyfit(range(len(sentiment_score)), sentiment_score, degree)
        poly_sentiment = np.poly1d(coeffs_sentiment)
        plt.plot(time, poly_sentiment(range(len(time))), color='blue', linestyle='--', label=f'Cumulative Return Polynomial Approx (degree {degree})')

        # Add labels, legend, and title
        plt.xlabel('Time Period', fontsize=12)
        plt.ylabel('Values', fontsize=12)
        plt.title(f' {speech_id} in {date}', fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)

        # Show plot
        plt.show()

def main(df, deltabefore=0, deltaafter=0, top_n=5, degree=2):
    """
    Perform all operations: calculate volatility, get top values, and plot results.

    Parameters:
    df : pandas.DataFrame
        The input dataframe containing all necessary data.
    deltabefore : int, optional
        Number of initial rows to exclude for volatility and plotting.
    deltaafter : int, optional
        Number of final rows to exclude for volatility and plotting.
    top_n : int, optional
        Number of top volatility values to select for plotting.
    degree : int, optional
        Degree of the polynomial approximation for plotting (default is 2).

    Returns:
    None
    """
    # Calculate volatility
    volatility = volatility_calculator(df, deltabefore=deltabefore, deltaafter=deltaafter)
    
    # Get top volatility values
    best_volatility = get_best_values(volatility, number=top_n)
    
    # Plot sentiment vs cumulative return
    plot_sentiment_vs_cumret(df, best_volatility, deltabefore=deltabefore, deltaafter=deltaafter, degree=degree)


# Example Usage
if __name__ == "__main__":
   main()
