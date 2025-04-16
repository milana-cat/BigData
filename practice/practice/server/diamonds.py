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

def plot_decision_boundary(model, X, y, title):
    h = 0.1  # шаг сетки
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFAAFF', '#FFFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#FFFF00']) 

    # Границы графика
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Предсказание для каждой точки сетки
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Визуализация
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.savefig(f"./{model}_boundary.png")
    plt.close()
    return f"./{model}_boundary.png"

class json_serialize(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def plot_regression(model, X, y,X_test,y_pred, title):
    plt.figure()
    sample_size = int(X.shape[0] * 1)
    X_sampled = X.sample(n=sample_size, random_state=42)
    y_sampled = y[X_sampled.index]
    
    plt.plot(X_test,y_pred,c='g',label='prediction')
    plt.scatter(X_sampled, y_sampled, c='k', label='data')
    plt.title(title)
    plt.xlabel('pH')
    plt.ylabel('quality')
    plt.axis('tight')
    plt.legend()
    plt.savefig(f"./{model}_regression.png")
    plt.close()
    return f"./{model}_regression.png"

def analyze_diamonds():
    # Загрузка данных
    file_path = "./WineQT.csv"
    data = pd.read_csv(file_path)
    data.drop('Id', axis=1, inplace=True)
    print(data.info())
    X = data
    print("1.Данные загружены успешно.")
    # 2. Разведочный анализ данных
    # 2.A. Количество строк и столбцов
    num_rows = X.shape[0]
    num_cols = X.shape[1]
    print("2.A. Количество строк:", num_rows)
    print("Количество столбцов:", num_cols)

    # 2.B. Объем памяти датафрейма
    memory_usage = X.memory_usage(deep=True).sum() / (1024 ** 2)
    print("\n2.B. Объем памяти датафрейма: {:.2f} МБ".format(memory_usage))

    # 2.C. Статистики для интервальных переменных
    num_cols = X.select_dtypes(include=['float64', 'int64']).columns

    print("\n2.C. Статистики для интервальных переменных:")
    num_stats={}
    for column in num_cols:
        min_value = X[column].min()
        mean_value = X[column].mean()
        median_value = X[column].median()
        max_value = X[column].max()
        percentile_25 = X[column].quantile(0.25)
        percentile_75 = X[column].quantile(0.75)
        print(f"{column}: min={min_value:.2f}, mean={mean_value:.2f}, median={median_value:.2f}, max={max_value:.2f}, 25%={percentile_25:.2f}, 75%={percentile_75:.2f}")
        num_stats[column] = {
            'min': min_value,
            'mean': mean_value,
            'median': median_value,
            'max': max_value,
            '25%': percentile_25,
            '75%': percentile_75
        }

    # 2.D. Анализ категориальных переменных
    print("\n2.D. Анализ категориальных переменных:")
    cat_stats = {}
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) == 0:
        print("Категориальных переменных не обнаружено.")
    else:
        for col in cat_cols:
            mode_value = X[col].mode()[0]
            mode_count = (X[col] == mode_value).sum()
            print(f"{col}: мода={mode_value}, частота={mode_count}")
            cat_stats[col] = {
            'mode': mode_value,
            'count': mode_count
            }

    # 3. Подготовка данных
    # 3.A. Пропуски
    print("\n3.A. Анализ пропусков:")
    if X.isnull().sum().sum() == 0:
        print("Пропуски отсутствуют.")
    else:
        print("Пропуски обнаружены. Заполняем средними значениями.")
        for column in num_cols:
            X[column] = X[column].fillna(0)
            print(f"Пропуски в {column} заполнены значением {0}")
        for column in cat_cols:
            X[column] = X[column].fillna("-")
            print(f"Пропуски в {column} заполнены значением '-'")

    # 3.B. Категориальные переменные
    print("\n3.b. Категориальные переменные:")
    if len(cat_cols) == 0:
        print("Категориальных переменных нет, кодирование не требуется.")
    else:
        le = LabelEncoder()
        for column in cat_cols:
            # Применяем Label Encoding
            X[f'{column}_Code'] = le.fit_transform(X[column])
            X.drop(column,axis=1,inplace=True)
            print(f"Столбец {column} заменён на {column}_Code")
        print("Категориальные переменные закодированы.")

    # 3.C. Выбросы
    print("\n3.C. Обработка выбросов:")
    for column in X.columns:
        Q1 = X[column].quantile(0.25)
        Q3 = X[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        X[column] = X[column].apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))
    print("Выбросы обработаны.")

    sX = X[['fixed acidity','density']].astype('float64')
    y = np.digitize(data["alcohol"],bins=[i for i in range(int(data["alcohol"].min()),int(data["alcohol"].max()),1)])
    # 3.D. Разделение данных
    print("\n3.D. Разделение данных на train и test:")
    X_train, X_test, y_train, y_test = train_test_split(sX, y, test_size=0.2, random_state=42)
    print(f"Train: {X_train.shape[0]} строк")
    print(f"Test: {X_test.shape[0]} строк")

    # 4. Обучение моделей
    models = {
        'KNN': KNeighborsClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=10000)
    }


    print("\n4. Результаты моделей:")
    results = {}
    classifier_plots = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred,average='micro'),
        }
        #print(f"{name}: Accuracy={results[name]['Accuracy']:.3f}, F1={results[name]['F1']:.3f}")
        print(f"{name}: Accuracy={results[name]['Accuracy']:.3f}, F1={results[name]['F1']:.3f}")
        classifier_plots.append(plot_decision_boundary(model, X_train, y_train, f"Decision Boundary ({name})"))

    # 5. Выбор лучшей модели
    best_model = max(results, key=lambda x: results[x]['F1'])
    print(f"\n5. Лучшая модель: {best_model} (F1={results[best_model]['F1']:.3f})")

    X=X.sort_values(by='fixed acidity')
    sX = X[['fixed acidity']]
    y = X['density']
    # 3.D. Разделение данных
    print("\n3.D. Разделение данных на train и test:")
    X_train, X_test, y_train, y_test = train_test_split(sX, y, test_size=0.2, random_state=42)
    print(f"Train: {X_train.shape[0]} строк")
    print(f"Test: {X_test.shape[0]} строк")

    # 4. Обучение моделей
    models = {
        'KNN': KNeighborsRegressor(),
        'Linear Regression': LinearRegression()
    }
    def mean_absolute_percentage_error(y_true, y_pred): 
        return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-10)) * 100)
                    
    print("\n4. Результаты моделей регрессии:")
    results_regression = {}
    regression_plots = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Расчет метрик
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results_regression[name] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }
        
        # Вывод результатов
        print(f"{name}:")
        print(f"  MAE: {mae:.2f}")
        print(f"  MSE: {mse:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  R²: {r2:.4f}\n")
        
        regression_plots.append(plot_regression(model, X_train, y_train, X_test, y_pred, f"Regression ({name})"))

    # 5. Выбор лучшей модели для регрессии (по R²)
    best_reg_model = max(results_regression, key=lambda x: results_regression[x]['R2'])
    print(f"\n5. Лучшая регрессионная модель: {best_reg_model} (R²={results_regression[best_reg_model]['R2']:.4f})")
    return json.dumps({
        'stats': {
            'num_rows': num_rows,
            'num_cols': len(num_cols),
            'memory_usage': memory_usage,
            'num_stats': num_stats,
            'cat_stats': cat_stats
        },
        'classification_metrics': results,
        'regression_metrics': results_regression,
        'decision_boundary_plots': classifier_plots,
        'regression_plots': regression_plots
    },cls=json_serialize)

analyze_diamonds()