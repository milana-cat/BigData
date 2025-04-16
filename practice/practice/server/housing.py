import pandas as pd
import numpy as np
import json
import h2o
from h2o.automl import H2OAutoML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from io import BytesIO
import base64

class json_serialize(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

import os

def analyze_housing():
    h2o.init()

    file_path = "./WineQT.csv"
    data = pd.read_csv(file_path)
    data.drop('Id', axis=1, inplace=True)
    X = data.copy()

    num_rows, num_cols = X.shape
    memory_usage = X.memory_usage(deep=True).sum() / (1024 ** 2)

    num_stats = {}
    num_columns = X.select_dtypes(include=['float64', 'int64']).columns
    for column in num_columns:
        num_stats[column] = {
            'min': X[column].min(),
            'mean': X[column].mean(),
            'median': X[column].median(),
            'max': X[column].max(),
            '25%': X[column].quantile(0.25),
            '75%': X[column].quantile(0.75)
        }

    cat_stats = {}
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        mode_value = X[col].mode()[0]
        mode_count = (X[col] == mode_value).sum()
        cat_stats[col] = {'mode': mode_value, 'count': mode_count}

    for column in num_columns:
        X[column].fillna(0, inplace=True)
    for column in cat_cols:
        X[column].fillna('-', inplace=True)

    le = LabelEncoder()
    for column in cat_cols:
        X[f'{column}_Code'] = le.fit_transform(X[column])
        X.drop(column, axis=1, inplace=True)

    for column in X.columns:
        Q1 = X[column].quantile(0.25)
        Q3 = X[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        X[column] = X[column].clip(lower, upper)

    X['fixed acidity'] = np.digitize(data['fixed acidity'], bins=list(range(int(data['fixed acidity'].min()), int(data['fixed acidity'].max()))))

    # Классификация
    cls_data = X.copy()
    train_cls, test_cls = train_test_split(cls_data, test_size=0.2, random_state=42)
    train_cls_h2o = h2o.H2OFrame(train_cls)
    test_cls_h2o = h2o.H2OFrame(test_cls)
    train_cls_h2o['fixed acidity'] = train_cls_h2o['fixed acidity'].asfactor()
    test_cls_h2o['fixed acidity'] = test_cls_h2o['fixed acidity'].asfactor()

    aml_cls = H2OAutoML(max_runtime_secs=60, seed=42)
    aml_cls.train(x=[col for col in train_cls.columns if col != 'fixed acidity'], y='fixed acidity', training_frame=train_cls_h2o)
    cls_perf = aml_cls.leader.model_performance(test_cls_h2o)
    f1_score = cls_perf.mean_per_class_error()

    # Регрессия
    reg_data = X.copy()
    train_reg, test_reg = train_test_split(reg_data, test_size=0.2, random_state=42)
    train_reg_h2o = h2o.H2OFrame(train_reg)
    test_reg_h2o = h2o.H2OFrame(test_reg)

    aml_reg = H2OAutoML(max_runtime_secs=60, seed=42)
    aml_reg.train(x=[col for col in train_reg.columns if col != 'density'], y='density', training_frame=train_reg_h2o)
    reg_perf = aml_reg.leader.model_performance(test_reg_h2o)
    regression_metrics = {
    'AutoML': {
        'MAE': reg_perf.mae(),
        'RMSE': reg_perf.rmse(),
        'R2': reg_perf.r2()
    }
}

    preds = aml_reg.leader.predict(test_reg_h2o).as_data_frame().values.flatten()
    actuals = test_reg['density'].values

    plt.figure(figsize=(8, 6))
    plt.scatter(actuals, preds, alpha=0.6)
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--')
    plt.xlabel("Actual Quality")
    plt.ylabel("Predicted Quality")
    plt.title("Regression: Actual vs Predicted")

    output_path = "./regression_plot.png"
    plt.savefig(output_path)
    plt.close()

    h2o.shutdown(prompt=False)

    return json.dumps({
        'stats': {
            'num_rows': num_rows,
            'num_cols': num_cols,
            'memory_usage': memory_usage,
            'num_stats': num_stats,
            'cat_stats': cat_stats
        },
        'classification_metrics': {
            'f1': f1_score
        },
        'regression_metrics': regression_metrics,
        'regression_plots': [output_path]
    }, cls=json_serialize)

analyze_housing()