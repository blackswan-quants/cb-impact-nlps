import helper_speech_analysis as hsa
import memory_handling as mh

#define variables
start_date = 2020
end_date = 2024
pickle_file_name = 'fedspeechees_sentiment_'

# create the aggregate dataframe
df = hsa.aggregate_sentiment_iterator(start_date, end_date, pickle_file_name)

#save file to csv for testing
df.to_csv('fedspeechees_sentiment_aggregate_'+str(start_date)+'_'+str(end_date)+'_', index=False)

# Save the final dataframe to a pickle file
pickle_helper = mh.PickleHelper(df)
pickle_helper.pickle_dump('fedspeechees_sentiment_aggregate_'+str(start_date)+'_'+str(end_date)+'_')

print(df)

