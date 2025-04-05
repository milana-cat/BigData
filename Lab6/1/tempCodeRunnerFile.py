# таблица самых частых слов
tfidf_df.to_csv("C:/Users/alexs/Downloads/BigData/BigData/Lab6/1/top_tfidf_words.csv", index=False)

# WordCloud — визуализация
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(zip(words, tfidf_scores)))
