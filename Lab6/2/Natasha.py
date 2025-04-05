import re
import pandas as pd
from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab, Doc
import nltk
from nltk.corpus import stopwords

# Настройка
nltk.download("stopwords")
stop_words = set(stopwords.words("russian"))

# Инициализация Natasha
segmenter = Segmenter()
morph_tagger = NewsMorphTagger(NewsEmbedding())
morph_vocab = MorphVocab()

# Предобработка
def preprocess_text(text):
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

# Загрузка стихов
with open("C:/Users/alexs/Downloads/BigData/BigData/Lab6/2/poems.txt", "r", encoding="utf-8") as f:
    raw_poems = f.read()

# Разделим по строкам или другим маркерам между стихами
poems = [s.strip() for s in raw_poems.split("\n\n") if s.strip()]

# Создаём DataFrame
df_poems = pd.DataFrame({"text": poems})
df_poems["label"] = 1  # 1 = стих
df_poems["processed_text"] = df_poems["text"].apply(preprocess_text)

# Сохраняем предобработанные стихи
df_poems.to_csv("C:/Users/alexs/Downloads/BigData/BigData/Lab6/2/processed_poems.csv", index=False)
