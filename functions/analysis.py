
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from wordcloud import WordCloud

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
    def calculate_volatility(group):
        # Ensure there are enough rows to apply deltabefore and deltaafter
        if len(group) > deltabefore + deltaafter:
            group_filtered = group.iloc[deltabefore: -deltaafter]  # Apply slicing to exclude rows
            
            
            # Calculate standard deviation (volatility)
            volatility = group_filtered['pct_change'].std()
            return volatility
        return None  # If not enough data, return None
    # Filter rows based on deltabefore and deltaafter

    volatility_series = df_prices_final.groupby(['title','date']).apply(calculate_volatility)

    
    # Reset index and rename column
    volatility_df = volatility_series.reset_index()
    volatility_df.columns.values[2] = 'volatility'
    volatility_df = volatility_df.dropna()

    # Merge with original dataframe to include 'date' and 'speech'
    final_df = pd.merge(df_prices_final[['date', 'title']].drop_duplicates(), 
                        volatility_df, 
                        on=['title','date'], 
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
    # Create a set of (title, date) pairs from df_top_values
    best_title_date_pairs = set(zip(df_top_values['title'], df_top_values['date']))

    # Filter df for rows where (title, date) matches the pairs in best_title_date_pairs
    df_filtered = df[df.apply(lambda row: (row['title'], row['date']) in best_title_date_pairs, axis=1)]

    for speech_id, speech_group in df_filtered.groupby(['title','date']):
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
                plt.plot(time[:deltabefore+1], pct_change[:deltabefore+1], color="lightblue", linewidth=0.75)
            plt.plot(time[deltabefore:-deltaafter], pct_change[deltabefore:-deltaafter], color="blue", linewidth=1.5)
            if deltaafter > 0:
                plt.plot(time[-deltaafter-1:], pct_change[-deltaafter-1:], color="lightblue", linewidth=0.75)
        else:
            if deltabefore > 0:
                plt.plot(time[:deltabefore+1], pct_change[:deltabefore+1], color="lightblue", linewidth=0.75)
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

def plot_vwap_by_speech(df, df_top_values, interval=5):
    """
    Plots the price, VWAP (Volume Weighted Average Price), VWAP bands, 
    and sentiment of an asset over time for each unique speech and date in the DataFrame.

    Args:
        df (pd.DataFrame): A DataFrame containing the following columns:
            - 'timestamp': Time data (timestamps).
            - 'close': Asset's price data.
            - 'volume': Asset's volume data.
            - 'sentiment': Sentiment values (can have missing values).
            - 'title': Speech identifier.
            - 'date': Date of the data.
        interval (int): The interval (in minutes) used to set the time axis tick spacing.

    Returns:
        None: The function plots the time series of the price, VWAP, VWAP bands, 
              and sentiment for each unique speech and date.
    """

    best_title_date_pairs = set(zip(df_top_values['title'], df_top_values['date']))

    # Filter df for rows where (title, date) matches the pairs in best_title_date_pairs
    df_filtered = df[df.apply(lambda row: (row['title'], row['date']) in best_title_date_pairs, axis=1)]
    df = df_filtered
    # Iterate over unique combinations of speech and date
    for speech, date in df[['title', 'date']].drop_duplicates().itertuples(index=False):
        # Filter the DataFrame for the specific speech and date
        df_filtered = df[(df['title'] == speech) & (df['date'] == date)]

        # Assign variables
        time = df_filtered['timestamp']
        prices = df_filtered['close']
        volumes = df_filtered['volume']
        sentiment = df_filtered['finbert_score']

        # Handle missing sentiment values (exclude from plot)
        sentiment_valid = sentiment.dropna()
        time_sentiment = time[sentiment_valid.index]

        # Calculate VWAP
        cumulative_price_volume = 0
        cumulative_volume = 0
        vwap_values = []
        volatility_factors = []

        for i in range(len(prices)):
            volatility_factors.append(prices[:i + 1].std())        
        for price, volume in zip(prices, volumes):
            cumulative_price_volume += price * volume
            cumulative_volume += volume
            if cumulative_volume == 0:
                vwap_values.append(None)  # Avoid division by zero
            else:
                vwap_values.append(cumulative_price_volume / cumulative_volume)

        # Create VWAP Bands
        upper_band_1 = [vwap + (1 * vol) for vwap, vol in zip(vwap_values, volatility_factors)]
        lower_band_1 = [vwap - (1 * vol) for vwap, vol in zip(vwap_values, volatility_factors)]

        upper_band_2 = [vwap + (1.5 * vol) for vwap, vol in zip(vwap_values, volatility_factors)]
        lower_band_2 = [vwap - (1.5 * vol) for vwap, vol in zip(vwap_values, volatility_factors)]

        # Plotting
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Set the x-axis locator and formatter for time intervals
        ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=interval))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        # Plot Price and VWAP
        ax1.plot(time, prices, color='blue', label='Price', linewidth=2)
        ax1.plot(time, vwap_values, color='green', label='VWAP', linewidth=2)

        # Plot the first set of bands
        ax1.fill_between(
            time, lower_band_1, upper_band_1, 
            color='lightgreen', alpha=0.4, label='VWAP Bands (\u00b11\u03c3)'
        )

        # Plot the second set of bands
        ax1.fill_between(
            time, lower_band_2, upper_band_2, 
            color='lightblue', alpha=0.3, label='VWAP Bands (\u00b11.5\u03c3)'
        )

        # Plot sentiment on a secondary y-axis
        ax2 = ax1.twinx()
        ax2.plot(time_sentiment, sentiment_valid, color='red', label='Sentiment', linewidth=2, linestyle='dashed')
        ax2.set_ylabel('Sentiment', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Title and labels
        ax1.set_title(f'Price vs VWAP with Bands and Sentiment ({speech} - {date})')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price / VWAP')

        # Rotate x-axis labels to make them readable
        plt.xticks(rotation=45)

        # Show the legends
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), bbox_transform=ax1.transAxes)

        # Display the plot
        plt.tight_layout()  # Adjust layout to avoid overlap
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

    #plot the VWAP bands to include volume information 
    plot_vwap_by_speech(df, best_volatility)

    #plot the wordcloud 
    plot_wordscloud(df, best_volatility)

# Example Usage
if __name__ == "__main__":
   main()
