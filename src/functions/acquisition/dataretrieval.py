"""
dataretrieval.py

This module fetches historical minute-level data for a given S&P500 ticker from the Twelve Data API.
It processes the data into a DataFrame containing [datetime, close, volume, timezone], handles pagination,
respects rate limits, and saves the resulting DataFrame as a pickle file.

Usage:
    python dataretrieval.py

Dependencies:
    - os
    - requests
    - pandas
    - pickle
    - python-dotenv (dotenv)
    - datetime (standard library)
    - time (standard library)
    - tqdm
    - logging
"""

import os
import requests
import pandas as pd
import pickle
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import logging


def datafetch(ticker_name: str) -> pd.DataFrame:
    """
    Fetch historical minute-level data for a given ticker from the Twelve Data API.

    This function handles pagination to cover the period from 2020-01-01 to 2024-12-31 (or up to the current date),
    respects API rate limits, logs progress and errors, and finally saves the data as a pickle file.

    Parameters
    ----------
    ticker_name : str
        The ticker symbol of the S&P 500 company (e.g., 'SPY').

    Returns
    -------
    pd.DataFrame
        DataFrame containing columns: ['datetime', 'close', 'volume', 'timezone'].
    """
    # Configure logging
    logging.basicConfig(
        filename='../datafetch.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load API key from environment file
    load_dotenv('../api.env')
    API_KEY = os.getenv('API_KEY')
    if not API_KEY:
        logging.error("API_KEY not found. Please ensure it's set in the api.env file.")
        raise ValueError("API_KEY not found. Please ensure it's set in the api.env file.")

    # Define API endpoint and parameters
    endpoint = "https://api.twelvedata.com/time_series"
    interval = '1min'
    timezone_param = 'America/New_York'
    outputsize = 5000

    # Define the date range
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 12, 31)
    current_datetime = datetime.now()
    if end_date > current_datetime:
        logging.info(f"Adjusting end_date from {end_date.date()} to current date {current_datetime.date()}.")
        end_date = current_datetime

    # Prepare to fetch data in batches
    all_data = []
    current_start = start_date
    minutes_per_day = 390  # Approximate number of trading minutes per day
    max_days_per_request = int(outputsize / minutes_per_day)
    total_days = (end_date - start_date).days + 1
    total_requests = (total_days // max_days_per_request) + 1

    logging.info(f"Fetching minute-level data for {ticker_name} from {start_date.date()} to {end_date.date()}.")

    # Rate limiting: 7.5 seconds per call (60/8)
    rate_limit_interval = 60 / 8
    last_api_call_time = None

    with tqdm(total=total_requests, desc="Fetching data") as pbar:
        while current_start <= end_date:
            days_remaining = (end_date - current_start).days + 1
            days_to_fetch = min(days_remaining, max_days_per_request)
            current_end = current_start + timedelta(days=days_to_fetch - 1)

            # Format dates as strings
            current_start_str = current_start.strftime("%Y-%m-%d")
            current_end_str = current_end.strftime("%Y-%m-%d")

            params = {
                'symbol': ticker_name,
                'interval': interval,
                'start_date': current_start_str,
                'end_date': current_end_str,
                'apikey': API_KEY,
                'timezone': timezone_param,
                'format': 'JSON',
                'outputsize': outputsize
            }

            # Enforce rate limiting
            if last_api_call_time:
                elapsed_time = time.time() - last_api_call_time
                if elapsed_time < rate_limit_interval:
                    sleep_time = rate_limit_interval - elapsed_time
                    logging.info(f"Sleeping for {sleep_time:.2f} seconds to respect rate limits.")
                    time.sleep(sleep_time)

            # Fetch data with retry logic
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    response = requests.get(endpoint, params=params, timeout=30)
                    if response.status_code == 200:
                        break
                    elif response.status_code == 429:
                        wait_time = 60  # Wait if rate limited
                        logging.warning(f"Rate limit exceeded. Waiting for {wait_time} seconds.")
                        time.sleep(wait_time)
                    else:
                        logging.warning(f"Attempt {attempt} failed with status {response.status_code}. Retrying...")
                        time.sleep(5)
                except requests.exceptions.Timeout:
                    logging.warning(f"Attempt {attempt} timed out. Retrying...")
                    time.sleep(5)
            else:
                logging.error(
                    f"Failed to fetch data for {ticker_name} from {current_start_str} to {current_end_str} after {max_retries} attempts.")
                raise ConnectionError(f"Failed to fetch data after {max_retries} attempts.")

            last_api_call_time = time.time()
            data = response.json()

            # Check for API errors
            if 'code' in data:
                error_code = data.get('code')
                error_message = data.get('message', 'No message provided.')
                if error_code == 400 and "No data is available on the specified dates" in error_message:
                    logging.warning(
                        f"No data available for {ticker_name} between {current_start_str} and {current_end_str}. Skipping this date range.")
                    pbar.update(1)
                    current_start = current_end + timedelta(days=1)
                    continue
                else:
                    logging.error(f"API Error {error_code}: {error_message}")
                    raise ValueError(f"API Error {error_code}: {error_message}")

            timezone = data.get('meta', {}).get('timezone', timezone_param)
            timeseries = data.get('values', [])
            if not timeseries:
                logging.warning(f"No data found for {ticker_name} between {current_start_str} and {current_end_str}.")
                pbar.update(1)
                current_start = current_end + timedelta(days=1)
                continue

            # Process each entry from the response
            for entry in timeseries:
                datetime_str = entry.get('datetime')
                if datetime_str:
                    try:
                        dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        dt = None
                    all_data.append({
                        'datetime': dt,
                        'close': float(entry.get('close', 0)),
                        'volume': int(entry.get('volume', 0)),
                        'timezone': timezone
                    })
                else:
                    all_data.append({
                        'datetime': None,
                        'close': None,
                        'volume': None,
                        'timezone': timezone
                    })

            logging.info(f"Fetched data from {current_start_str} to {current_end_str}.")
            pbar.update(1)
            current_start = current_end + timedelta(days=1)

    # Create DataFrame from collected data and drop rows with missing datetime
    df = pd.DataFrame(all_data)
    df.dropna(inplace=True)

    # Define output pickle file path
    pickle_folder = "data/pickle_files"
    pickle_filename = f"{ticker_name}_2024_data.pkl"
    if not os.path.exists(pickle_folder):
        os.makedirs(pickle_folder)
    pickle_filepath = os.path.join(pickle_folder, pickle_filename)

    # Save DataFrame as pickle
    with open(pickle_filepath, 'wb') as file:
        pickle.dump(df, file)

    logging.info(f"Data for {ticker_name} saved to {pickle_filepath}")
    print(f"Data for {ticker_name} saved to {pickle_filepath}")

    return df


def main():
    """
    Main entry point for data retrieval. Fetch data for the ticker 'SPY'.
    """
    df = datafetch('SPY')
    print(df.head())


if __name__ == '__main__':
    main()
