import re
import pandas as pd
from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab, Doc
import nltk
from nltk.corpus import stopwords

# 1. Скачиваем стоп-слова
nltk.download("stopwords")
stop_words = set(stopwords.words("russian"))
stop_words.add("это")

# 2. Natasha инициализация
segmenter = Segmenter()
morph_tagger = NewsMorphTagger(NewsEmbedding())
morph_vocab = MorphVocab()

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

# 3. Загрузка текстов песен
with open("C:/Users/alexs/Downloads/BigData/BigData/Lab6/1/songs.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 4. Разделение на песни
songs = re.split(r"\n\d+\.\s*", raw_text)
songs = [s.strip() for s in songs if s.strip()]
df = pd.DataFrame({"song_text": songs})
df["processed_text"] = df["song_text"].apply(preprocess_natasha)

# 5. Сохраняем результат
df.to_csv(r"C:/Users/alexs/Downloads/BigData/BigData/Lab6/1/processed_songs.csv", index=False)
