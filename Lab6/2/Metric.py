from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import pandas as pd
from gensim.models import Word2Vec
import numpy as np

# Загрузка
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

# Получение средних векторов для каждого текста
def get_avg_vector(words, model, vector_size):
    vectors = [model.wv[word] for word in words if word in model.wv]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vector_size)

X = np.array([get_avg_vector(words, model, 100) for words in tokenized_texts])
y = labels.values


# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Список моделей
models = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "SVC": SVC(kernel='rbf'),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000)
}

# Обучение и оценка
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n {name} ")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=1))
