import pandas as pd
import sys
import os

# Add the parent directory of the current script to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from helpermodules import memory_handling as mh

# Initialize an empty list to hold all the dataframes
df_list = []

# List of years for which you have pickled files
years = ['2020', '2021', '2022', '2023', '2024']

# Loop through the years and load each pickled file
for year in years:
    file = f"{year}sentiment.pkl"
    helper = mh.PickleHelper.pickle_load(file)  # Assuming this loads the dataframe
    df = helper.obj  # Extract the dataframe from the loaded object
    df_list.append(df)  # Add the dataframe to the list

# Concatenate all dataframes in the list to create one final dataframe
df_sentiment_final = pd.concat(df_list, ignore_index=True)

df_sentiment_final.to_csv('2020-2024sentiment.csv', index=False)
pickle_helper = mh.PickleHelper(df_sentiment_final)
pickle_helper.pickle_dump('2020-2024sentiment')
