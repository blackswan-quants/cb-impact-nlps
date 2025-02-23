import pandas as pd
from libs.helpermodules import memory_handling as mh
df = pd.read_csv('/Users/baudotedua/Dropbox/Mac/Documents/GitHub/cb-impact-nlps/csv_files/2020-2024prices.csv')

timezone = df.timezone.unique()[0]
df = df.drop(columns='timezone')
df['datetime'] = pd.to_datetime(df['datetime'])
df['datetime'] = df['datetime'].dt.tz_localize(timezone)

pickle_helper = mh.PickleHelper(df)
pickle_helper.pickle_dump('2020-2024prices')