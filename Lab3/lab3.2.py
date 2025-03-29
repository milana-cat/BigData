import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn import neighbors, datasets
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC


def load_data():
   #Загрузка данных из базы данных в DataFrame.
    try:
        df = pd.read_csv('train.csv')
        print("Данные успешно загружены в DataFrame")
        return df
    except Exception as e:
        print("Ошибка выполнения запроса:", e)
        exit()

def analyze_dataset(df):
    #Подсчёт строк и столбцов.
    rows, columns = df.shape
    print(f"Количество строк: {rows}, Количество столбцов: {columns}")

def analyze_memory(df):
    df.info(memory_usage='deep')

def exploratory_analysis_numeric(df, columns):
    #Разведочный анализ числовых переменных.
    for col in columns:
        try:
            print(f"\nАнализ числовой переменной: {col}")
            print(f"Доля пропусков: {df[col].isnull().mean():.4%}")
            print(f"Максимум: {df[col].max()}")
            print(f"Минимум: {df[col].min()}")
            print(f"Среднее значение: {df[col].mean():.4f}")
            print(f"Медиана: {df[col].median():.4f}")
            print(f"Перцентиль 25: {df[col].quantile(0.25):.4f}")
            print(f"Перцентиль 75: {df[col].quantile(0.75):.4f}")
        except Exception as e:
            print(f"Ошибка анализа переменной {col}: {e}")

def exploratory_analysis_categorical(df):
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_columns:
        print(f"\nАнализ категориальной переменной: {col}")
        print(f"Количество уникальных значений: {df[col].nunique()}")
        print(f"Мода: {df[col].mode()[0] if not df[col].mode().empty else 'Нет моды'}")
        print(f"Мода (количество значений): {df[col].mode().value_counts() if not df[col].mode().empty else 'Нет моды'}")

def fill_NaN(df, columns):
    for col in columns:
        df[col].fillna(df[col].mean(), inplace=True)
##Провести анализ и обработку пропусков (либо заменить, либо удалить)
def correlation_table(df, target_column):
    #Построение таблицы корреляции.
    df = df.drop('id', axis=1)
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    #numeric_columns.delete("id")
    #del df["id"]
    correlation = df[numeric_columns].corr()[target_column].sort_values(ascending=False)
    print("\nТаблица корреляции с целевым столбцом:")
    print(correlation)
    # Построение тепловой карты корреляций
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Корреляционная матрица")
    plt.show()

n_neighbors = 15
df =load_data()
analyze_dataset(df)
analyze_memory(df)
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
exploratory_analysis_numeric(df, numeric_columns)
exploratory_analysis_categorical(df)
correlation_table(df, target_column="target")
fill_NaN(df, numeric_columns)

# Рассчитаем межквартильный диапазон
Q1 = df['cond'].quantile(0.25)
Q3 = df['cond'].quantile(0.75)
IQR = Q3 - Q1

# Определение границ для выбросов
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Удаление строк с выбросами
data = df[(df['cond'] > lower_bound) & (df['cond'] < upper_bound)]


# Удаляем строки с пропущенными значениями в целевой переменной
data = df.dropna(subset=["cond"])
# C. Обработка категориальных переменных
encoder = OneHotEncoder()
categorical_columns = df.select_dtypes(include=['object', 'category']).columns
encoded_data = encoder.fit_transform(df[categorical_columns])
# Выбор признаков и целевой переменной
X = df[["cond"]]  # Используем квадратные скобки для выбора столбца   Все столбцы
y = df["target"]  # Целевая переменная

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Нормализация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Подбор гиперпараметра k с использованием перекрёстной проверки
best_k = 1
best_score = 0

for k in range(1, 21):  # Пробуем значения k от 1 до 20
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)  # 5-кратная перекрёстная проверка
    if scores.mean() > best_score:
         best_k = k
         best_score = scores.mean()

print(f"Оптимальное значение k: {best_k}")

 # Обучение модели с оптимальным k
knn = neighbors.KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

 # Предсказания на тестовой выборке
y_pred = knn.predict(X_test)
y_pred_proba = knn.predict_proba(X_test)[:, 1]  # Для ROC-AUC
# Оценка качества
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
f1 = f1_score(y_test, y_pred, average='micro')
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
print("Confusion Matrix:\n", conf_matrix)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Логистическая регрессия
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_predictions = log_reg.predict(X_test)
log_reg_accuracy = accuracy_score(y_test, log_reg_predictions)
print("Logistic Regression")
y_pred_proba = log_reg.predict_proba(X_test)[:, 1]  # Для ROC-AUC
# Оценка качества
conf_matrix = confusion_matrix(y_test, log_reg_predictions)
accuracy = accuracy_score(y_test, log_reg_predictions)
precision = precision_score(y_test, log_reg_predictions, average='micro')
recall = recall_score(y_test,log_reg_predictions, average='micro')
f1 = f1_score(y_test, log_reg_predictions, average='micro')
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
print("Confusion Matrix:\n", conf_matrix)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")



# SVM
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
svm_predictions = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print("SVM ")

# Оценка качества
conf_matrix = confusion_matrix(y_test, svm_predictions)
accuracy = accuracy_score(y_test, svm_predictions)
precision = precision_score(y_test, svm_predictions, average='micro')
recall = recall_score(y_test,svm_predictions, average='micro')
f1 = f1_score(y_test, svm_predictions, average='micro')

print("Confusion Matrix:\n", conf_matrix)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
