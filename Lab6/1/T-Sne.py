import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import Word2Vec

# Загружаем модель Word2Vec и TF-IDF таблицу
model = Word2Vec.load("C:/Users/alexs/Downloads/BigData/BigData/Lab6/1/song_word2vec.model")
df_tfidf = pd.read_csv("C:/Users/alexs/Downloads/BigData/BigData/Lab6/1/top_tfidf_words.csv")

# Получаем 15 самых значимых слов
top_words = df_tfidf["word"].head(30).tolist()  # берём 30, чтобы отфильтровать ниже

# Фильтруем: только те слова, которые есть в словаре модели
filtered_words = [word for word in top_words if word in model.wv]
filtered_vectors = np.array([model.wv[word] for word in filtered_words])

# Применяем t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=5, n_iter=1000)
vectors_2d = tsne.fit_transform(filtered_vectors)

# Строим график
plt.figure(figsize=(10, 6))
for i, word in enumerate(filtered_words):
    x, y = vectors_2d[i]
    plt.scatter(x, y)
    plt.annotate(word, (x + 0.5, y + 0.5), fontsize=10)

plt.title("t-SNE визуализация 15 часто встречающихся слов (Word2Vec)")
plt.grid(True)
plt.tight_layout()
plt.savefig("C:/Users/alexs/Downloads/BigData/BigData/Lab6/1/tsne_top15_words.png")
plt.show()
