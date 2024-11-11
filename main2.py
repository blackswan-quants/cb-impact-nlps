import pickle

df = open ("./pickle_files/fedspeeches_preprocessed.pkl", "rb")
df = pickle.load(df)
print(df)