import pandas as pd
from helpermodules import memory_handling as mh
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from functions import compute_sentiment, filtering_df, retrieve_datas, scraping_speeches, update_realtime, analysis
from functions.update_realtime import change_time
from functions.filtering_df import main as filtering
from functions.analysis import main as plot
from functions.scraping_speeches import main as scraping

#######
'''questa prima parte del codice presenta il codice che dovrebbe essere implementato 
(quello preceduto da '#') e in seguito il codice che ho utlizzato io non avendo ancora 
accesso alle funzioni, riciclando i file csv e 'pulendoli' per la task.
Le funzioni che verranno implementate devono restituire un oggetto che abbia 
la stessa forma e caratteristiche dell'oggetto finale (evidenziato nel codice)'''

yearlist = [2020,2021,2022,2023,2024]
# df_fed = scraping(yearlist) -> ALREADY RUN, STORE IN ANOTHER FILE
file = "2020-2024fedspeeches.pkl"
helper = mh.PickleHelper.pickle_load(file)
df_fed = helper.obj


#df_prices = retrieve_datas(df_speech, deltabefore, deltaafter) -> ALREADY RUN
file2 = "2020-2024prices.pkl"
helper = mh.PickleHelper.pickle_load(file2)
df = helper.obj

print(df.columns) 
    
df['datetime'] = df.apply(
    lambda row: row['datetime'].tz_localize('UTC').tz_convert(row['timezone']), axis=1
)
# Drop the 'timezone' column as it's no longer needed
df = df.drop(columns=['timezone'])
df['date'] = df['datetime'].dt.date
df['datetime'] = df['datetime'] + pd.Timedelta(hours=4) # updated to the correct value
df_prices = df #<--- OGGETTO FINALE


#df_speech = computespeech()
file3 = "2020-2024speeches.pkl"
helper = mh.PickleHelper.pickle_load(file3)
df_speech = helper.obj

df_speech = df_speech[df_speech['date']>='2020-01-01']
df_speech = df_speech.sort_values(['date','timestamp'], ascending=True) #<--- OGGETTO FINALE


#df_sentiment = compute_sentiment()
file4 = "2020-2024sentiment.pkl"
helper = mh.PickleHelper.pickle_load(file4)
df_sentiment = helper.obj 


############

df_fed.rename(columns={'timestamp': 'opening_time'}, inplace=True)
#update the correct timestamp for df_speech
df_speech_final = change_time(df_speech, df_fed)
#sorting the values
df_speech_final = df_speech_final.sort_values(['date','timestamp'], ascending=True)

###

# Define market open and close times (in ET)
market_open = pd.to_datetime('09:30:00', format='%H:%M:%S').time()
market_close = pd.to_datetime('16:00:00', format='%H:%M:%S').time()

# Extract time from 'timestamp'
df_speech_final['time'] = df_speech_final['timestamp'].dt.time

# Filter rows where 'time' is within market hours
df_filtered = df_speech_final[(df_speech_final['time'] >= market_open) & (df_speech_final['time'] < market_close)]
df_speech_final = df_speech_final.drop(columns=['time'])

###

df_speech=df_speech_final




#combining together all the values to have a final dataframe including datas for 
#speech, sentiment, pct_change
#select a timedelta to also show prior and successive pct_change values 
deltabefore = 5
deltaafter = 5
df_speech_final, df_prices_final = filtering(df_prices, df_speech, df_sentiment,deltabefore, deltaafter)

#plot the best top_n values for volatility over the speech time 
plot(df_speech_final, deltabefore, deltaafter, top_n=5)

# TO FIX -> it still fucks up some graphs -> should be fixed 

# TO FIX -> SI PERDONO L'80% DELLE SPEECHES TRA QUELLE TROVATE DA LORENZO E QUELLE 
# TROVATE SUL SITO DELLA FED -> NUMERO MOLTO ALTO 

# -> fix the delta function DONE 
# -> do some volume studying -> VWAP bands graph DONE
# -> add the function to save the words graph (visualization purpose) DONE
# -> do some correlation analysis (t:t), (t-1:t) ADD THE FUNCTION 
# -> extend the studying up to 2020 values DONE 
