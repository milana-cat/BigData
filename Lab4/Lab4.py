import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind, pearsonr

import joblib


def save_model(model, filename):
    """Сохраняет модель в файл с использованием joblib."""
    joblib.dump(model, filename)
    print(f"Модель сохранена в {filename}")

def load_model(filename):
    """Загружает модель из файла с использованием joblib."""
    model = joblib.load(filename)
    print(f"Модель загружена из {filename}")
    return model

def load_data(file_path):
    """Загружает данные из CSV-файла."""
    return pd.read_csv(file_path)

def data_overview(df):
    """Выводит информацию о количестве строк и столбцов, а также о размере в памяти."""
    num_rows, num_cols = df.shape
    memory_usage = df.memory_usage(deep=True).sum() / 1024 ** 2  # В мегабайтах
    print(f"Количество строк: {num_rows}")
    print(f"Количество столбцов: {num_cols}")
    print(f"Общий объем памяти: {memory_usage:.2f} MB")

def numeric_summary(df):
    """Выводит статистики для числовых переменных (мин, медиана, среднее, макс, персентили 25 и 75)."""
    summary = df.describe(percentiles=[0.25, 0.75]).T[['min', '50%', 'mean', 'max', '25%', '75%']]
    summary.rename(columns={'50%': 'median'}, inplace=True)
    print("Статистика числовых переменных:")
    print(summary)

def categorical_summary(df):
    """Выводит моду и количество ее повторений для категориальных переменных."""
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) == 0:
        print("В наборе данных нет категориальных переменных.")
    else:
        for col in categorical_cols:
            mode_value = df[col].mode()[0]
            mode_count = df[col].value_counts().iloc[0]
            print(f"Переменная: {col}")
            print(f"Мода: {mode_value}, встречается {mode_count} раз\n")

def handle_missing_values(df):
    """Обрабатывает пропущенные значения (заполняет медианой)."""
    df.fillna(df.median(numeric_only=True), inplace=True)
    print("Пропущенные значения обработаны (заполнены медианой).")

def handle_outliers(df):
    """Обрабатывает выбросы с использованием межквартильного размаха (IQR)."""
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower_bound, upper_bound)
    print("Выбросы обработаны с использованием метода IQR.")

def encode_categorical(df):
    """Кодирует категориальные переменные с помощью One-Hot Encoding."""
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        print("Категориальные переменные закодированы методом One-Hot Encoding.")
    return df


def test_hypotheses(df):
    """Проверяет две гипотезы на данных."""
    print("\nГипотеза 1: Стоимость перерабатываемых упаковок ниже, чем у неперерабатываемых")
    diabetic = df[df['recyclable_package'] == 1]['cost']
    non_diabetic = df[df['recyclable_package'] == 0]['cost']
    t_stat, p_value = ttest_ind(diabetic, non_diabetic)
    print(f"t-статистика: {t_stat:.2f}, p-значение: {p_value:.5f}")
    print("Гипотеза отвергается" if p_value < 0.05 else "Гипотеза не отвергается")

    print("\nГипотеза 2: Корреляция между Продажах в магазинах и стоимостью.")
    correlation, p_value = pearsonr(df['store_sales(in millions)'], df['cost'])
    print(f"Коэффициент корреляции: {correlation:.2f}, p-значение: {p_value:.5f}")
    print("Корреляция значима" if p_value < 0.05 else "Корреляция не значима")
    
    
def split_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)  # Применяем тот же scaler, что был обучен на X_train

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test), list(X.columns)

def correlation_analysis(df, target_column):
    #Построение таблицы корреляции признаков и целевого столбца.
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    correlation_table = df[numeric_columns].corr()[target_column].sort_values(ascending=False)
    print("\nТаблица корреляции с целевым столбцом:")
    print(correlation_table)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_columns].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Корреляционная матрица')
    plt.show()
class CustomKNNRegressor:
    """Реализация алгоритма KNN для регрессии."""
    def __init__(self, n_neighbors=24):
        self.n_neighbors = n_neighbors
    
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
    
    def predict(self, X):
        X = np.array(X)
        y_pred = []
        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            neighbors_idx = np.argsort(distances)[:self.n_neighbors]
            y_pred.append(np.mean(self.y_train[neighbors_idx]))
        return np.array(y_pred)


class CustomLinearRegression:
    def __init__(self, learning_rate=0.001, n_iterations=5000, l2_penalty=0, clip_value=1000.0):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l2_penalty = l2_penalty  # Регуляризация L2
        self.clip_value = clip_value  # Ограничение градиента

    def fit(self, X, y, batch_size=8):
        X = np.array(X)
        y = np.array(y)
        X = np.c_[np.ones(X.shape[0]), X]  # Добавляем столбец единиц для свободного члена
        self.theta = np.zeros(X.shape[1])

        for i in range(self.n_iterations):
            y_pred = X @ self.theta
            error = y - y_pred

            # Вычисление градиента с учетом L2-регуляризации
            gradients = (-2 / X.shape[0]) * (X.T @ error) + (self.l2_penalty / X.shape[0]) * self.theta
            gradients = np.clip(gradients, -self.clip_value, self.clip_value)  # Ограничение градиента
            self.theta -= self.learning_rate * gradients

            # Проверка на NaN и бесконечность
            if np.isnan(self.theta).any() or np.isinf(self.theta).any():
                print(f"Градиентный спуск не сходится, остановка обучения на {i}-й итерации.")
                break


    def predict(self, X):
        X = np.array(X)
        X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.theta



class ElasticNet:
    def __init__(self, alpha=0, l1_ratio=1, learning_rate=0.1, epochs=5000):
        self.alpha = alpha  # Параметр регуляризации
        self.l1_ratio = l1_ratio  # Доля L1 регуляризации (0.5 означает смесь Lasso и Ridge)
        self.learning_rate = learning_rate  # Скорость обучения
        self.epochs = epochs  # Количество итераций

    def fit(self, X, y):
        # Инициализация коэффициентов
        self.coef_ = np.zeros(X.shape[1])
        m = len(y)
        
        # Градиентный спуск
        for epoch in range(self.epochs):
            predictions = self.predict(X)
            errors = predictions - y
            
            # Градиент по ошибке
            gradient = (1/m) * X.T.dot(errors)

            # Градиент для L2 регуляризации
            gradient += self.alpha * (1 - self.l1_ratio) * self.coef_

            # Градиент для L1 регуляризации (сигнум — это знак коэффициента)
            gradient += self.alpha * self.l1_ratio * np.sign(self.coef_)
            
            # Обновляем коэффициенты
            self.coef_ -= self.learning_rate * gradient

    def predict(self, X):
        return X.dot(self.coef_)

    def get_coefficients(self):
        return self.coef_

def train_models(X_train, y_train):
    """Обучает кастомные модели KNN и линейной регрессии."""
    models = {
        'Custom KNN': CustomKNNRegressor(n_neighbors=5),
        'Custom ElasticNet': ElasticNet(),
        'Custom Linear Regression':CustomLinearRegression()
    }
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models

def evaluate_models(models, X_test, y_test):
    """Оценивает модели с использованием метрик MAE, MSE, RMSE, MAPE и R^2."""
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Убираем нулевые значения перед расчетом MAPE
        non_zero_mask = y_test != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((y_test[non_zero_mask] - y_pred[non_zero_mask]) / y_test[non_zero_mask])) * 100
        else:
            mape = np.nan  # Если все значения нулевые, выставляем NaN
        
        r2 = r2_score(y_test, y_pred)
        results[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'R^2': r2}
    return results


def plot_predictions(model, X_test, y_test, model_name="Linear Regression"):
    """
    Визуализация предсказанных значений относительно фактических.
    """
    # Предсказание
    y_pred = model.predict(X_test)

    # Построение графика
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, color='black', label="data", alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="идеальное предсказание")

    plt.xlabel("Фактические значения", fontsize=12)
    plt.ylabel("Предсказанные значения", fontsize=12)
    plt.title(f"График предсказаний для {model_name}", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


    
if __name__ == "__main__":
    file_path = "train.csv"  
    df_reduced = load_data(file_path)
    df = df_reduced.sample(frac=0.03 , random_state=42)
    df_columns = df.columns.tolist()
    print(df_columns)
    data_overview(df)
    numeric_summary(df)
    categorical_summary(df)
    correlation_analysis(df, "cost")
    handle_missing_values(df)
    handle_outliers(df)
    df = encode_categorical(df)
    test_hypotheses(df)
    target_column = "store_sales(in millions)"  # Целевая переменная   поменять местами
    feature_column = ["unit_sales(in millions)", "florist"] # Несколько столбцов подавать на вход.
    X_train, X_test, y_train, y_test, feature_names = split_data(df, target_column)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n")

    trained_models = train_models(X_train_scaled, y_train)
    model_results = evaluate_models(trained_models, X_test_scaled, y_test )
    
    for model, metrics in model_results.items():
        print(f"\n{model}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    # Выводим список доступных признаков
    print("Доступные признаки:", feature_names)
 
    model = trained_models["Custom Linear Regression"]  # Берем именно обученную модель
    # Вызов функции
    plot_predictions(model, X_test_scaled, y_test, "Custom Linear Regression")
    # Сохранение модели
    save_model(trained_models["Custom ElasticNet"], "custom_ElasticNet_regression.pkl")
    
    # Загрузка модели
    #loaded_model = load_model("custom_linear_regression.pkl")

    # Проверка работы загруженной модели
    #y_pred = loaded_model.predict(X_test)

    # Визуализация предсказаний загруженной модели
    #plot_predictions(loaded_model, X_test, y_test, "Loaded Custom Linear Regression")

   
