import pandas as pd
from gensim.models import Word2Vec

# 9. Загрузка предобработанных текстов
df = pd.read_csv("C:/Users/alexs/Downloads/BigData/BigData/Lab6/1/processed_songs.csv")

# Подготовка токенизированных данных
tokenized_texts = [text.split() for text in df["processed_text"]]

# 10. Обучение модели Word2Vec
model = Word2Vec(
    sentences=tokenized_texts,
    vector_size=100,   # размерность векторов слов
    window=5,          # окно контекста
    min_count=2,       # минимальное число вхождений слова
    workers=4,         # количество потоков
    sg=1               # 1 — skip-gram, 0 — CBOW
)

# Сохраняем модель (опционально)
model.save("C:/Users/alexs/Downloads/BigData/BigData/Lab6/1/song_word2vec.model")

# 11. Проверка: близкие слова к заданному
try:
    similar_words = model.wv.most_similar("пустота", topn=10)
    print("Ближайшие слова к 'пустота':")
    for word, similarity in similar_words:
        print(f"{word}: {similarity:.3f}")
except KeyError:
    print("Слово 'пустота' не найдено в словаре модели. Попробуй другое.")
