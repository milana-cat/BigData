import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab, Doc
import nltk
from nltk.corpus import stopwords

# Инициализация
nltk.download("stopwords")
stop_words = set(stopwords.words("russian"))
stop_words.add("это")
segmenter = Segmenter()
morph_tagger = NewsMorphTagger(NewsEmbedding())
morph_vocab = MorphVocab()

# Предобработка текста
def preprocess_natasha(text):
    text = text.lower()
    text = re.sub(r"[^а-яё\s]", " ", text)
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    lemmas = []
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
        lemma = token.lemma
        if len(lemma) > 2 and lemma not in stop_words:
            lemmas.append(lemma)
    return " ".join(lemmas)

# Интерфейс Streamlit
st.title("Анализ текста статьи о процессорах")
st.subheader("Загрузка и предобработка текста")

# Загружаем текст
with open("CPU№03.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Обработка
processed_text = preprocess_natasha(raw_text)
st.text_area("Предобработанный текст", processed_text[:1000] + "...")

# TF-IDF
st.subheader("TF-IDF и облако слов")
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform([processed_text])
words = vectorizer.get_feature_names_out()
scores = X.toarray().flatten()
tfidf_df = pd.DataFrame({"word": words, "tfidf": scores})
tfidf_df = tfidf_df.sort_values(by="tfidf", ascending=False)

# Облако слов
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(zip(words, scores)))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
st.pyplot(plt)

# Показываем топ-слов
st.write("Топ-15 слов по TF-IDF:")
st.dataframe(tfidf_df.head(15))

# Word2Vec
st.subheader("Word2Vec + t-SNE визуализация")
tokens = [processed_text.split()]
model = Word2Vec(sentences=tokens, vector_size=100, window=5, min_count=1, sg=1)
top_words = tfidf_df.head(30)["word"].tolist()
filtered_words = [w for w in top_words if w in model.wv]
vectors = np.array([model.wv[word] for word in filtered_words])

# t-SNE
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
tsne_result = tsne.fit_transform(vectors)

# Визуализация
plt.figure(figsize=(10, 6))
for i, word in enumerate(filtered_words):
    x, y = tsne_result[i]
    plt.scatter(x, y)
    plt.annotate(word, (x + 0.5, y + 0.5))
plt.title("t-SNE визуализация частотных слов")
plt.grid(True)
st.pyplot(plt)

try:
# Получение похожих слов
    similar_words = model.wv.most_similar("процессор", topn=10)
    words = [word for word, _ in similar_words]
    similarities = [similarity for _, similarity in similar_words]
    vectors = np.array([model.wv[word] for word in words])

    # Таблица
    df = pd.DataFrame(similar_words, columns=["Слово", "Сходство"])
    st.subheader("Похожие слова к 'процессор'")
    st.dataframe(df)


except KeyError:
    print("Слово 'пустота' не найдено в словаре модели. Попробуй другое.")