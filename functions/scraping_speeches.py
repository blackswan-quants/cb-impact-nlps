# import functions
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
from helpermodules import memory_handling as mh

def breakdown_html(url):
    # Example URL

    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)

    # Regex for time validation
    time_pattern = re.compile(r'^\d{1,2}:\d{2} (a\.m\.|p\.m\.)$')

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all "row cal-nojs__rowTitle" sections
        sections = soup.find_all('div', class_='row cal-nojs__rowTitle')
        
        # Initialize lists for storing data
        titles_list = []
        times_list = []
        dates_list = []

        for section in sections:
            # Find the <h4> header within the section
            header = section.find('h4', class_='col-md-12')

            if header:
                header_text = header.get_text(strip=True)
                

                # Extract data only for "Speeches" 
                if header_text =="Speeches":
                    print(f"Extracting data for {header_text}...")

                    # Collect all following `div` elements with class "row"
                    current = section.find_next_sibling('div', class_='row')

                    while current and not current.find('h4', class_='col-md-12'):
                        # Extract title, time, and date elements
                        title = current.find('div', class_='col-xs-7')
                        time = current.find('div', class_='col-xs-2')
                        date = current.find('div', class_='col-xs-3')

                        # Append data to lists if found
                        if title:
                            titles_list.append(title)
                        if time:
                            times_list.append(time.get_text(strip=True))
                        if date:
                            dates_list.append(date.get_text(strip=True))

                        # Move to the next sibling
                        current = current.find_next_sibling('div', class_='row')

        # Output the lists
        #VERY PRONE TO COMMIT MISTAKES
        titles_list=titles_list[0:]
        times_list=times_list[1:]
        dates_list=dates_list[1:]
        #print("Titles List:", titles_list)
        #print("Times List:", times_list)
        #print("Dates List:", dates_list)
    else:
        print(f"Failed to fetch the page: {response.status_code}")
    return titles_list, dates_list, times_list


def handle_titles(titles_list):
    speaker_names = []
    calendar_titles = []

    for tag in titles_list:
        # Extract the first <p> tag for the speaker's name
        speaker_tag = tag.find('p')
        if speaker_tag:
            # Extract the name part (assuming the format "Discussion -- Speaker Name")
            name_text = speaker_tag.get_text(strip=True)
            # Split the name from the rest of the text
            speaker_name = name_text.split('--')[1].strip() if '--' in name_text else name_text
            speaker_names.append(speaker_name)
        else:
            speaker_names.append(None)

        # Extract the <p class="calendar__title"> for the title, looking for <em>
        title_tag = tag.find('p', class_='calendar__title')
        if title_tag and title_tag.find('em'):
            calendar_titles.append(title_tag.find('em').get_text(strip=True))
        else:
            calendar_titles.append(None)

    return speaker_names, calendar_titles


def time_handling(times_list):
    """
    Convert a list of time strings in various formats (e.g., '1:00 p.m.') 
    to a standardized time format 'H:M:S'.

    Args:
        times_list (list): List of time strings in formats like '1:00 p.m.', '09:00 am'.

    Returns:
        list: List of times in 'H:M:S' format.
    """
    updated_times = []
    for time_str in times_list:
        try:
            # Preprocess the input to remove unnecessary periods
            clean_time_str = time_str.replace('.', '').strip()
            # Parse the cleaned time string
            parsed_time = datetime.strptime(clean_time_str, '%I:%M %p')
            # Convert to time format H:M:S and append to the result list
            updated_times.append(parsed_time.strftime('%H:%M:%S'))
        except ValueError as e:
            print(f"Error parsing time string: {time_str}. Ensure it follows formats like '1:00 p.m.' or '09:00 am'.")
            raise e
    return updated_times



def handle_dates(days_list, month, year):
    """
    Convert a list of days, a month, and a year into a list of datetime.date objects.

    Args:
        days_list (list): List of numbers or strings representing the days of the month.
        month (str): Month as a string (e.g., 'January', 'February').
        year (int): Year as a number (e.g., 2024).

    Returns:
        list: List of datetime.date objects.
    """
    # Convert month name to its corresponding number
    try:
        month_number = datetime.strptime(month, '%B').month
    except ValueError:
        raise ValueError(f"Invalid month name: '{month}'. Use the full month name (e.g., 'January').")
    
    # Ensure all days in the list are integers
    try:
        days_list = [int(day) for day in days_list]
    except ValueError:
        raise ValueError("All elements in days_list must be integers or convertible to integers.")
    
    # Generate datetime.date objects for each day
    dates_list = []
    for day in days_list:
        try:
            # Create the date object
            date_obj = date(year, month_number, day)
            dates_list.append(date_obj)
        except ValueError:
            print(f"Invalid date: Year={year}, Month={month_number}, Day={day}.")
            raise
    
    return dates_list


def create_dataframe( titles_list, dates_list, times_list, month, year):
    speaker_names , speech_titles = handle_titles(titles_list)
    times = time_handling(times_list)
    date = handle_dates(dates_list, month, year)

    # Creating a dictionary with list values
    data = {'date': date, 'speaker': speaker_names, 'title': speech_titles, 'timestamp':times}

    # Creating the DataFrame
    df = pd.DataFrame(data)
    return df


def remove_time_from_datetime(df):
    # Ensure the 'date' column is of datetime type
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Now remove the time part (normalize) while keeping the 'date' as a datetime object
    df['date'] = df['date'].dt.normalize()

    return df


def add_timezone(df):
    # Define the Eastern Time zone
    eastern = pytz.timezone('US/Eastern')

    # Function to ensure that 'date' and 'timestamp' are datetime objects and handle them
    def process_row(row):
        try:
            # Ensure both 'date' and 'timestamp' are datetime objects
            date = pd.to_datetime(row['date'], format='%y:%m:%d')  # Ensure 'date' is in datetime format
            timestamp = pd.to_datetime(row['timestamp'], format='%H:%M:%S').time()  # Ensure 'timestamp' is in datetime format
            # Combine and localize the datetime
            combined_datetime = datetime.combine(date, timestamp)
            localized_time = eastern.localize(combined_datetime, is_dst=None)
            final_timestamp = localized_time.strftime('%H:%M:%S%:z')  # Keep only time and timezone info
        except Exception as e:
            print(f"Error processing row: {row}. Error: {e}")
            final_timestamp = None  # Return None or any default value
        return final_timestamp

        # Step 6: Apply the function to the whole dataset
    df['timestamp'] = df.apply(process_row, axis=1)
        
    return df

def main(yearlist):
    #list month- year 
    
    months = ['january', 'february', 'march', 'april', 'may', 'june', 
          'july', 'august', 'september', 'october', 'november', 'december']
    host = 'www.federalreserve.gov'
    prefix = '/newsevents/'
    suffix = '.htm'
    final_combined_df = pd.DataFrame()
    for year in yearlist:
        for month in months:
            mid_str = f"{year}-{month}"
            url  = 'https://' + host + prefix + mid_str + suffix
            print('processing datas for',month,year,'\n')
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