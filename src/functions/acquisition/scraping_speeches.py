"""
scraping_speeches.py

This module scrapes Federal Reserve speech details from the Federal Reserve website.
It extracts speech titles, speaker names, dates, and times from HTML pages and
saves the aggregated data to both CSV and pickle formats.

Usage:
    python scraping_speeches.py

Dependencies:
    - pandas
    - numpy
    - bs4 (BeautifulSoup)
    - requests
    - pytz
    - datetime
    - re
    - utils.memory_handling (local module)
"""

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from http.client import HTTPSConnection
import pickle
import pytz
from urllib.request import urlopen
import requests
import os
from datetime import datetime, date
import re
from utils import memory_handling as mh


def breakdown_html(url: str) -> tuple[list, list, list]:
    """
    Scrapes the given URL and extracts lists of title sections, dates, and times.

    Parameters
    ----------
    url : str
        The URL to scrape.

    Returns
    -------
    tuple of lists
        A tuple containing lists of titles, dates, and times extracted from the HTML.
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)

    # Regex for time validation (if needed)
    time_pattern = re.compile(r'^\d{1,2}:\d{2} (a\.m\.|p\.m\.)$')

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        sections = soup.find_all('div', class_='row cal-nojs__rowTitle')

        titles_list = []
        times_list = []
        dates_list = []

        for section in sections:
            header = section.find('h4', class_='col-md-12')
            if header:
                header_text = header.get_text(strip=True)
                if header_text == "Speeches":
                    print(f"Extracting data for {header_text}...")
                    current = section.find_next_sibling('div', class_='row')
                    while current and not current.find('h4', class_='col-md-12'):
                        title = current.find('div', class_='col-xs-7')
                        time_div = current.find('div', class_='col-xs-2')
                        date_div = current.find('div', class_='col-xs-3')
                        if title:
                            titles_list.append(title)
                        if time_div:
                            times_list.append(time_div.get_text(strip=True))
                        if date_div:
                            dates_list.append(date_div.get_text(strip=True))
                        current = current.find_next_sibling('div', class_='row')

        # Adjust lists if needed
        titles_list = titles_list[:]  # Could add additional slicing logic here if needed
        times_list = times_list[1:]
        dates_list = dates_list[1:]
    else:
        print(f"Failed to fetch the page: {response.status_code}")
        titles_list, dates_list, times_list = [], [], []

    return titles_list, dates_list, times_list


def handle_titles(titles_list: list) -> tuple[list, list]:
    """
    Extract speaker names and speech titles from HTML tags.

    Parameters
    ----------
    titles_list : list
        List of BeautifulSoup tag objects containing title info.

    Returns
    -------
    tuple of lists
        Two lists: speaker_names and calendar_titles.
    """
    speaker_names = []
    calendar_titles = []

    for tag in titles_list:
        speaker_tag = tag.find('p')
        if speaker_tag:
            name_text = speaker_tag.get_text(strip=True)
            speaker_name = name_text.split('--')[1].strip() if '--' in name_text else name_text
            speaker_names.append(speaker_name)
        else:
            speaker_names.append(None)

        title_tag = tag.find('p', class_='calendar__title')
        if title_tag and title_tag.find('em'):
            calendar_titles.append(title_tag.find('em').get_text(strip=True))
        else:
            calendar_titles.append(None)

    return speaker_names, calendar_titles


def time_handling(times_list: list) -> list:
    """
    Convert a list of time strings to standardized 'H:M:S' format.

    Parameters
    ----------
    times_list : list
        List of time strings (e.g., '1:00 p.m.').

    Returns
    -------
    list
        List of time strings in 'H:M:S' format.
    """
    updated_times = []
    for time_str in times_list:
        try:
            clean_time_str = time_str.replace('.', '').strip()
            parsed_time = datetime.strptime(clean_time_str, '%I:%M %p')
            updated_times.append(parsed_time.strftime('%H:%M:%S'))
        except ValueError as e:
            print(f"Error parsing time string: {time_str}. Ensure format like '1:00 p.m.' is used.")
            raise e
    return updated_times


def handle_dates(days_list: list, month: str, year: int) -> list:
    """
    Convert a list of days along with a month and year into datetime.date objects.

    Parameters
    ----------
    days_list : list
        List of day values (as numbers or strings).
    month : str
        Month name (e.g., 'January').
    year : int
        Year (e.g., 2024).

    Returns
    -------
    list
        List of datetime.date objects.
    """
    try:
        month_number = datetime.strptime(month, '%B').month
    except ValueError:
        raise ValueError(f"Invalid month name: '{month}'. Use full month name (e.g., 'January').")

    try:
        days_list = [int(day) for day in days_list]
    except ValueError:
        raise ValueError("All elements in days_list must be integers or convertible to integers.")

    dates_list = []
    for day in days_list:
        try:
            date_obj = date(year, month_number, day)
            dates_list.append(date_obj)
        except ValueError:
            print(f"Invalid date: Year={year}, Month={month_number}, Day={day}.")
            raise
    return dates_list


def create_dataframe(titles_list: list, dates_list: list, times_list: list, month: str, year: int) -> pd.DataFrame:
    """
    Create a DataFrame from scraped title, date, and time lists.

    Parameters
    ----------
    titles_list : list
        List of title tags.
    dates_list : list
        List of date strings.
    times_list : list
        List of time strings.
    month : str
        Month for the dates.
    year : int
        Year for the dates.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: 'date', 'speaker', 'title', 'timestamp'.
    """
    speaker_names, speech_titles = handle_titles(titles_list)
    times = time_handling(times_list)
    date_objs = handle_dates(dates_list, month, year)

    data = {'date': date_objs, 'speaker': speaker_names, 'title': speech_titles, 'timestamp': times}
    df = pd.DataFrame(data)
    return df


def remove_time_from_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the 'date' column to remove any time component.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'date' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with the 'date' column normalized.
    """
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['date'] = df['date'].dt.normalize()
    return df


def add_timezone(df: pd.DataFrame) -> pd.DataFrame:
    """
    Localize datetime entries in the 'timestamp' column to US Eastern Time.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'date' and 'timestamp' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'timestamp' adjusted to include timezone information.
    """
    eastern = pytz.timezone('US/Eastern')

    def process_row(row):
        try:
            date_obj = pd.to_datetime(row['date'])
            time_obj = pd.to_datetime(row['timestamp'], format='%H:%M:%S').time()
            combined = datetime.combine(date_obj, time_obj)
            localized = eastern.localize(combined, is_dst=None)
            return localized.strftime('%H:%M:%S%:z')
        except Exception as e:
            print(f"Error processing row: {row}. Error: {e}")
            return None

    df['timestamp'] = df.apply(process_row, axis=1)
    return df


def main(yearlist: list = None):
    """
    Main function to scrape and process Federal Reserve speech data over a range of years.

    Parameters
    ----------
    yearlist : list, optional
        List of years to process (default processes all available years in the data).

    The function iterates over all months for each year, scrapes speech data from the Federal Reserve website,
    aggregates the results into a single DataFrame, and saves the final dataset as CSV and pickle files.
    """
    if yearlist is None:
        yearlist = [2020, 2021, 2022, 2023, 2024]

    months = ['january', 'february', 'march', 'april', 'may', 'june',
              'july', 'august', 'september', 'october', 'november', 'december']
    host = 'www.federalreserve.gov'
    prefix = '/newsevents/'
    suffix = '.htm'
    final_combined_df = pd.DataFrame()

    for year in yearlist:
        for month in months:
            mid_str = f"{year}-{month}"
            url = f"https://{host}{prefix}{mid_str}{suffix}"
            print(f"Processing data for {month.capitalize()} {year}...\n")
            titles_list, dates_list, times_list = breakdown_html(url)
            final_df = create_dataframe(titles_list, dates_list, times_list, month, year)
            ultimate_df = remove_time_from_datetime(final_df)
            ultimate_df = add_timezone(ultimate_df)
            final_combined_df = pd.concat([final_combined_df, ultimate_df], ignore_index=True)

    print(final_combined_df)
    final_combined_df.to_csv('2020-2024speeches.csv', index=False)
    pickle_helper = mh.PickleHelper(final_combined_df)
    pickle_helper.pickle_dump('2020-2024fedspeeches')


if __name__ == "__main__":
    main()
