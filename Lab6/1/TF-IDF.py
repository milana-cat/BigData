import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/alexs/Downloads/BigData/BigData/Lab6/1/processed_songs.csv")

# TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["processed_text"])

# Получим сумму TF-IDF по каждому слову
tfidf_scores = X.sum(axis=0).A1
words = vectorizer.get_feature_names_out()

# Создаём DataFrame со словами и их суммарным TF-IDF
tfidf_df = pd.DataFrame({"word": words, "tfidf": tfidf_scores})
tfidf_df = tfidf_df.sort_values(by="tfidf", ascending=False)

# таблица самых частых слов
tfidf_df.to_csv("C:/Users/alexs/Downloads/BigData/BigData/Lab6/1/top_tfidf_words.csv", index=False)

# WordCloud — визуализация
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(zip(words, tfidf_scores)))

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Облако слов на основе TF-IDF")
plt.show()
