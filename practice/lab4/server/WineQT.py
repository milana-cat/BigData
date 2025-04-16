import pandas as pd
import os
import time
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier, LazyRegressor

def analyze_wine_data():
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "WineQT.csv")
    data = pd.read_csv(file_path)

    # Классификация: Квантильная разбивка по sulphates
    data['sulphates_class'] = pd.qcut(data['sulphates'], q=3, labels=["Low", "Medium", "High"])

    # === КЛАССИФИКАЦИЯ ===
    print("\n=== КЛАССИФИКАЦИЯ ===")
    classification_data = data[['pH', 'quality', 'sulphates_class']].copy()
    X_class = classification_data.drop(columns='sulphates_class')
    y_class = classification_data['sulphates_class']

    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
        X_class, y_class, test_size=0.2, random_state=42
    )

    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    start = time.time()
    models_class, predictions_class = clf.fit(X_train_class, X_test_class, y_train_class, y_test_class)
    print("Время обучения классификации:", round(time.time() - start, 2), "сек")
    print(models_class)

    # === РЕГРЕССИЯ ===
    print("\n=== РЕГРЕССИЯ ===")
    regression_data = data[['quality', 'sulphates']].copy()
    X_reg = regression_data.drop(columns='sulphates')
    y_reg = regression_data['sulphates']

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
    start = time.time()
    models_reg, predictions_reg = reg.fit(X_train_reg, X_test_reg, y_train_reg, y_test_reg)
    print("Время обучения регрессии:", round(time.time() - start, 2), "сек")
    print(models_reg)


analyze_wine_data()
