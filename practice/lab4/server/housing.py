import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from WineQT import json_serialize, plot_decision_boundary, plot_regression

# изменения внутри функции analyze_housing
def analyze_housing():
    # 1. Загрузка данных
    file_path = "./WineQT.csv"
    data = pd.read_csv(file_path)
    print(data.info())
    X = data.copy()
    print("1.Данные загружены успешно.")

    # 2. Анализ
    num_rows = X.shape[0]
    num_col_names = X.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns

    memory_usage = X.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"\n2.B. Объем памяти датафрейма: {memory_usage:.2f} МБ")

    # Статистики
    num_stats = {}
    for column in num_col_names:
        num_stats[column] = {
            'min': X[column].min(),
            'mean': X[column].mean(),
            'median': X[column].median(),
            'max': X[column].max(),
            '25%': X[column].quantile(0.25),
            '75%': X[column].quantile(0.75)
        }

    # Категориальные переменные
    cat_stats = {}
    if len(cat_cols) > 0:
        for col in cat_cols:
            cat_stats[col] = {
                'mode': X[col].mode()[0],
                'count': X[col].value_counts().iloc[0]
            }

    # 3. Подготовка
    if X.isnull().sum().sum() > 0:
        for column in num_col_names:
            X[column] = X[column].fillna(X[column].mean())
        for column in cat_cols:
            X[column] = X[column].fillna("-")

    le = LabelEncoder()
    for column in cat_cols:
        X[f'{column}_Code'] = le.fit_transform(X[column])
        X.drop(column, axis=1, inplace=True)

    # Обработка выбросов
    for column in num_col_names:
        Q1 = X[column].quantile(0.25)
        Q3 = X[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        X[column] = X[column].clip(lower=lower_bound, upper=upper_bound)

    # 4. Классификация (предсказание pH по quality + pH)
    sX = X[['quality', 'alcohol']]
    y = pd.cut(X["pH"], bins=3, labels=[0, 1, 2]).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(sX, y, test_size=0.2, random_state=42)

    models = {
        'KNN': KNeighborsClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=10000)
    }
    results = {}
    classifier_plots = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred, average='micro'),
        }
        classifier_plots.append(plot_decision_boundary(model, X_train, y_train, f"{name}_boundary"))

    # 5. Регрессия: предсказание pH по alcohol
    sX = X[['alcohol']]
    y = X['pH']
    X_train, X_test, y_train, y_test = train_test_split(sX, y, test_size=0.2, random_state=42)

    reg_models = {
        'KNN': KNeighborsRegressor(),
        'Linear Regression': LinearRegression()
    }
    def mean_absolute_percentage_error(y_true, y_pred): 
        return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-10)) * 100)

    results_regression = {}
    regression_plots = []
    for name, model in reg_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results_regression[name] = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAPE': mean_absolute_percentage_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }
        regression_plots.append(plot_regression(model, X_train, y_train, X_test, y_pred, f"{name}_regression"))

    return json.dumps({
        'stats': {
            'num_rows': num_rows,
            'num_cols': len(num_col_names),
            'memory_usage': memory_usage,
            'num_stats': num_stats,
            'cat_stats': cat_stats
        },
        'classification_metrics': results,
        'regression_metrics': results_regression,
        'decision_boundary_plots': classifier_plots,
        'regression_plots': regression_plots
    }, cls=json_serialize)
