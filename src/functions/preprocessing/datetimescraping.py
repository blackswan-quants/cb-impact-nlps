"""
datetimescraping.py

This module scrapes event date/time information from the Federal Reserve website.
It provides functions to:
  - Generate URLs for each month of a given year.
  - Parse time strings (e.g. '2:45 p.m.') into timezone‐aware datetime objects.
  - Scrape the Federal Reserve calendar page for event details.
  - Aggregate events across all months for a given year into a DataFrame.

Usage:
    python datetimescraping.py

Dependencies:
    - requests
    - bs4 (BeautifulSoup)
    - csv (standard library)
    - re
    - pandas
    - datetime
    - dateutil
    - pytz
    - time
    - random
"""

import requests
from bs4 import BeautifulSoup
import csv
import re
import pandas as pd
from datetime import datetime, date
from dateutil import parser
import pytz
import time
import random


def generate_month_urls(year: int) -> list[str]:
    """
    Generate URLs for each month of the specified year.

    Parameters
    ----------
    year : int
        The year for which to generate URLs.

    Returns
    -------
    list[str]
        A list of URLs for each month in the format:
        'https://www.federalreserve.gov/newsevents/YYYY-month.htm'
    """
    months = [
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december'
    ]
    urls = [f'https://www.federalreserve.gov/newsevents/{year}-{month}.htm' for month in months]
    return urls


def parse_time_to_datetime(timestamp_str: str, year: int, month: int) -> pd.Timestamp | None:
    """
    Parse a time string (e.g., '2:45 p.m.') into a timezone-aware datetime object.

    Since the date is not provided in the time string, this function assumes events occur
    on the 1st day of the given month.

    Parameters
    ----------
    timestamp_str : str
        A time string like '2:45 p.m.'.
    year : int
        The year to use.
    month : int
        The month (as a number) to use.

    Returns
    -------
    pd.Timestamp or None
        A pandas Timestamp localized to US/Eastern, or None if parsing fails.
    """
    if timestamp_str == 'N/A':
        return None

    try:
        # Construct a full date string with a fixed day (1st)
        date_str = f"{year}-{month:02d}-01 {timestamp_str}"
        dt = parser.parse(date_str)
        eastern = pytz.timezone('US/Eastern')
        dt = eastern.localize(dt)
        return pd.to_datetime(dt)
    except Exception as e:
        print(f"Error parsing time: {timestamp_str} with error {e}")
        return None


def scrape_federal_reserve_calendar(url: str) -> list[dict]:
    """
    Scrape a Federal Reserve calendar page for event details.

    The function extracts event titles and start times from the page.

    Parameters
    ----------
    url : str
        The URL of the Federal Reserve calendar page.

    Returns
    -------
    list[dict]
        A list of dictionaries, each with keys 'Title' and 'Start Time'.
    """
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/58.0.3029.110 Safari/537.3'
        )
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the page: {e}")
        return []

    soup = BeautifulSoup(response.text, 'lxml')
    if not soup:
        print("Failed to parse HTML content.")
        return []

    event_containers = soup.find_all('div', class_=re.compile(r'\bpanel\s+panel-unstyled\b'))
    print(f"Found {len(event_containers)} event containers for {url}.")

    if not event_containers:
        print("No event containers found. Verify HTML structure or CSS classes.")
        return []

    # Extract year and month from the URL. Format: /newsevents/YYYY-month.htm
    match = re.search(r'/newsevents/(\d{4})-([a-z]+)\.htm', url)
    if match:
        year = int(match.group(1))
        month_str = match.group(2)
        try:
            month_num = datetime.strptime(month_str.capitalize(), '%B').month
        except ValueError:
            month_num = 1
    else:
        year = 2024
        month_num = 1

    events = []
    for container in event_containers:
        # Extract time
        timestamp_div = container.find('div', class_=re.compile(r'\bcol-xs-2\b'))
        timestamp = timestamp_div.find('p').get_text(strip=True) if timestamp_div and timestamp_div.find('p') else 'N/A'
        # Extract title
        title_p = container.find('p', class_=re.compile(r'\bcalendar__title\b'))
        title = title_p.find('em').get_text(strip=True) if title_p and title_p.find('em') else 'N/A'
        if title == 'N/A':
            continue

        start_time = parse_time_to_datetime(timestamp, year, month_num)
        if start_time is None:
            continue

        events.append({
            'Title': title,
            'Start Time': start_time
        })

    return events


def scrape_all_months(year: int) -> pd.DataFrame:
    """
    Scrape Federal Reserve speech events for all months in a given year.

    Parameters
    ----------
    year : int
        The year to scrape (e.g., 2024).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing columns ['Title', 'Start Time'] for all events.
    """
    urls = generate_month_urls(year)
    all_events: list[dict] = []
    for url in urls:
        monthly_events = scrape_federal_reserve_calendar(url)
        if monthly_events:
            all_events.extend(monthly_events)
        time.sleep(random.uniform(1, 3))  # Sleep to avoid overwhelming the server

    df = pd.DataFrame(all_events, columns=['Title', 'Start Time'])
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


if __name__ == "__main__":
    year = 2024
    df = scrape_all_months(year)
    print(df)
    output_csv = f'federal_reserve_speech_events_{year}.csv'
    df.to_csv(output_csv, index=False)
    print(f"Data saved to '{output_csv}'")
