df_poems = pd.read_csv("C:/Users/alexs/Downloads/BigData/BigData/Lab6/2/processed_poems.csv")
df_songs = pd.read_csv("C:/Users/alexs/Downloads/BigData/BigData/Lab6/1/processed_songs.csv")
df_poems["label"] = 1
df_songs["label"] = 0

df_all = pd.concat([df_songs, df_poems], ignore_index=True)
texts = df_all["text"].fillna("").astype(str)
labels = df_all["label"]

# Токенизация
tokenized_texts = [text.split() for text in texts]

# Обучение Word2Vec
model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=2, workers=4, sg=1)
model.save("C:/Users/alexs/Downloads/BigData/BigData/Lab6/2/lyrics_word2vec.model")
