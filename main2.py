import pickle

df = open ("./pickle_files/fedspeeches_preprocessed.pkl", "rb")
df = pickle.load(df)

unique_speakers = df["speaker"].unique().tolist()
print(unique_speakers)