import pandas as pd
from helpermodules import memory_handling as mh
import numpy as np
import matplotlib.pyplot as plt
from functions import compute_sentiment, filtering_df, retrieve_datas, scraping_speeches, update_realtime, analysis
from functions.update_realtime import change_time
from functions.filtering_df import main as filtering
from functions.analysis import main as plot

#######
'''questa prima parte del codice presenta il codice che dovrebbe essere implementato 
(quello preceduto da '#') e in seguito il codice che ho utlizzato io non avendo ancora 
accesso alle funzioni, riciclando i file csv e 'pulendoli' per la task.
Le funzioni che verranno implementate devono restituire un oggetto che abbia 
la stessa forma e caratteristiche dell'oggetto finale (evidenziato nel codice)'''


#df_fed = scraping_speech(yearlist)
df_fed = pd.read_csv('2024speeches.csv')


#df_prices = retrieve_datas(df_speech, deltabefore, deltaafter)
df = pd.read_csv("/Users/baudotedua/Dropbox/Mac/Documents/GitHub/cb-impact-nlps/US SPX 500 (Mini) 1 Minute (1).csv")
columns_to_keep = ['<Date>', ' <Time>', ' <Open>', ' <Close>', ' <TotalVolume>']
df = df[columns_to_keep]
df.columns = ['date', 'time', 'open', 'close', 'volume']
print(df.columns) 
    # combining date and time in one column (format datetime)
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%d/%m/%Y %H:%M:%S')
print(df['datetime'].dtype)
df = df.drop(columns=['time'])
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
df = df[['datetime'] + [col for col in df.columns if col != 'datetime']]
df['datetime'] = df['datetime'].dt.tz_localize('America/New_York')
df_prices = df #<--- OGGETTO FINALE


#df_speech = computespeech()
file = "fedspeeches_preprocessed.pkl"
helper = mh.PickleHelper.pickle_load(file)
df_speech = helper.obj
df_speech = df_speech[df_speech['date']>='2024-01-01']
df_speech = df_speech.sort_values(['date','timestamp'], ascending=True) #<--- OGGETTO FINALE


#df_sentiment = compute_sentiment()
file = "fedspeechees_sentiment_2024.pkl"
helper = mh.PickleHelper.pickle_load(file)
df_sentiment = helper.obj #<----- OGGETTO FINALE

############

df_fed.rename(columns={'timestamp': 'opening_time'}, inplace=True)
#update the correct timestamp for df_speech
df_speech_final = change_time(df_speech, df_fed)
#sorting the values
df_speech_final = df_speech_final.sort_values(['date','timestamp'], ascending=True)
df_speech=df_speech_final


deltabefore = 5
deltaafter = 5
#combining together all the values to have a final dataframe including datas for 
#speech, sentiment, pct_change
df_speech_final, df_prices_final = filtering(df_prices, df_speech, df_sentiment, deltabefore, deltaafter)

#plot the best top_n values for volatility over the speech time 
plot(df_speech_final,deltabefore, deltaafter, top_n=3)