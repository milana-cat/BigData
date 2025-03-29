import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.model_selection import cross_val_score
from scipy.stats import ttest_ind
import matplotlib.ticker as mtick
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler,OneHotEncoder

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Шаг 1: Загрузка данных
# Замените 'your_dataset.csv' на путь к вашему файлу с данными
data = pd.read_csv('milknew.csv')
print(data.columns)
# Шаг 2: Предобработка данных

data = data[data['Grade'].isin(['medium', 'low'])]  # Фильтрация по целевой переменной
data['grade_numeric'] = data['Grade'].map({'low': 0, 'medium': 1})




numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
eda_results = {}
column = 'grade_numeric'
eda_results[column] = {
        'Доля пропусков': data[column].isnull().mean(),
        'Максимальное': data[column].max(),
        'Минимальное': data[column].min(),
        'Среднее': data[column].mean(),
        'Медиана': data[column].median(),
        'Дисперсия': data[column].var(),
        'Квантиль 0.1': data[column].quantile(0.1),
        'Квантиль 0.9': data[column].quantile(0.9),
        'Квартиль 1': data[column].quantile(0.25),
        'Квартиль 3': data[column].quantile(0.75)
    }


# Визуализация результатов EDA
eda_df = pd.DataFrame(eda_results).T
print(eda_df)

print("\nГипотеза 1: Молоко с оценкой Low имеет отличную температуру от молока с оценкой Medium")
low_temp = data[data['grade_numeric'] == 0]['Temprature'].dropna()
medium_temp = data[data['grade_numeric'] == 1]['Temprature'].dropna()
stat, p_value = ttest_ind(low_temp, medium_temp, equal_var=False)
print(f"t-test: статистика = {stat:.2f}, p-значение = {p_value:.2e}")

print("\nГипотеза 2: Молоко с оценкой Low имеет отличный цвет от молока с оценкой Medium?")
low_colour = data[data['grade_numeric'] == 0]['Colour'].dropna()
medium_colour = data[data['grade_numeric'] == 1]['Colour'].dropna()
stat, p_value = ttest_ind(low_colour, medium_colour, equal_var=False)
print(f"t-test: статистика = {stat:.2f}, p-значение = {p_value:.2e}")

# Посмотреть распределение целевой переменной
plt.figure(figsize=(10, 5))
sns.countplot(x='Grade', data=data, palette='viridis')
plt.title('Распределение значений grade')
plt.xlabel('Grade')
plt.ylabel('Количество')
plt.xticks(rotation=45)  # Поворот меток по оси x, если нужно
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(
        data=data,
        x="Temprature",
        y="pH",
        hue="Grade",
        alpha=0.7
    )
plt.title("")
plt.xlabel("Temprature")
plt.ylabel("pH")
ax = plt.gca()
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))  # Форматирование оси Y
plt.grid(True)
plt.legend(title="Зависимость класса молока от температуры и кислотности")
plt.show()




# Удаляем строки с пропущенными значениями в целевой переменной
data = data.dropna(subset=["Grade"])

# Преобразуем  в числовой формат
data['Grade'] = data['Grade'].map({'low': 0, 'medium': 1})

# Заполняем пропущенные значения в признаках (если есть) средним значением
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="mean")
data[["Temprature"]] = imputer.fit_transform(data[["Temprature"]])
# Выбор признаков и целевой переменной
X = data[["Temprature"]]  # Используем квадратные скобки для выбора столбца
y = data["Grade"]  # Целевая переменная

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
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)  # 5-кратная перекрёстная проверка
    if scores.mean() > best_score:
         best_k = k
         best_score = scores.mean()

print(f"Оптимальное значение k: {best_k}")

 # Обучение модели с оптимальным k
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

 # Предсказания на тестовой выборке
y_pred = knn.predict(X_test)
y_pred_proba = knn.predict_proba(X_test)[:, 1]  # Для ROC-AUC

# Оценка качества
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print("Confusion Matrix:\n", conf_matrix)
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"ROC-AUC: {roc_auc:.2f}")