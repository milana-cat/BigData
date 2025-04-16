import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from lightautoml.tasks import Task
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score



def analyze_wine_data():
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "WineQT.csv")
    data = pd.read_csv(file_path)

    # ➤ Статистика
    stats = {
        "num_rows": len(data),
        "num_cols": data.shape[1],
        "memory_usage": data.memory_usage(deep=True).sum() / 1e6,
        "num_stats": {},
        "cat_stats": {}
    }

    for col in data.select_dtypes(include='number').columns:
        stats["num_stats"][col] = {
            "min": data[col].min(),
            "mean": data[col].mean(),
            "median": data[col].median(),
            "max": data[col].max(),
            "25%": data[col].quantile(0.25),
            "75%": data[col].quantile(0.75),
        }

    for col in data.select_dtypes(include='object').columns:
        stats["cat_stats"][col] = {
            "mode": data[col].mode().iloc[0],
            "count": data[col].value_counts().iloc[0]
        }

    # ➤ Регрессия sulphates ~ quality
    X = data.drop(columns=['sulphates'])
    y = data['sulphates']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    automl = TabularAutoML(task=Task('reg'), timeout=60)
    oof_pred = automl.fit_predict(X_train, roles={"target": "sulphates"})
    test_pred = automl.predict(X_test)

    mae = mean_absolute_error(y_test, test_pred.data[:, 0])
    r2 = r2_score(y_test, test_pred.data[:, 0])

    regression_metrics = {
        "LightAutoML": {
            "MAE": mae,
            "R2": r2
        }
    }

    # ➤ Построение графика
    fig_path = os.path.join(current_dir, "regression_plot.png")
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_test, y=test_pred.data[:, 0])
    plt.xlabel("Фактические sulphates")
    plt.ylabel("Предсказанные sulphates")
    plt.title("Регрессия LightAutoML")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    return {
        "stats": stats,
        "regression_metrics": regression_metrics,
        "regression_plots": [fig_path]
    }
