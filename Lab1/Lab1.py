from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

# Данные для подключения
host = "povt-cluster.tstu.tver.ru"
port = 5432
user = "mpi"
password = "135a1"
database = "Criminal"

# Подключение к базе данных
try:
    engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{database}?client_encoding=utf8")
    print("Подключение успешно")
except Exception as e:
    print("Ошибка подключения:", e)
    exit()


# Проверка наличия таблицы и выполнения запроса
try:
    query = 'SELECT * FROM person JOIN get_fit_now_member ON person.id = get_fit_now_member.person_id JOIN drivers_license ON person.license_id = drivers_license.id JOIN income ON income.ssn = person.ssn  LIMIT 1000'  # Ограничиваем выборку
    df = pd.read_sql_query(query, engine)
    print("Данные успешно загружены в DataFrame")
    print("Колонки в таблице:", df.columns.tolist())  # Выводим список колонок для проверки
    print(df.head())

    # Фильтрация: выбор только важных признаков
    important_columns = ["age", "membership_status", "gender", "annual_income", "height"]
    filtered_df = df[important_columns]
    print("Отфильтрованный DataFrame:")
    print(filtered_df.head())

    # Одномерный анализ: Построение гистограмм
    for feature in ["age", "membership_status"]:  # Количественные признаки
        if feature in filtered_df.columns:
            plt.figure(figsize=(8, 5))
            plt.hist(filtered_df[feature].dropna(), bins=30, edgecolor="k", alpha=0.7)
            plt.title(f"Распределение признака: {feature}")
            plt.xlabel("Статус членства" if feature == "membership_status" else feature)
            plt.ylabel("Частота")

            plt.grid(True)
            plt.show()
        else:
            print(f"Признак '{feature}' не найден в данных.")

     # Многомерный анализ: Построение графиков

    # График 1: Зависимость saleprice от acreage без категорий SINGLE FAMILY
    filtered_data_no_single_family = filtered_df[filtered_df["membership_status"] != "regular"]

    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        data=filtered_data_no_single_family,
        x="height",
        y="age",
        hue="gender",
        alpha=0.7
    )
    plt.title("Зависимость роста от возраста без членства regular")
    plt.xlabel("Возраст")
    plt.ylabel("Рост")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))  # Форматирование оси Y
    plt.grid(True)
    plt.legend(title="Зависимость роста от возраста")
    plt.show()


    
    # График 2: Распределение saleprice по yearbuilt и soldasvacant
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=filtered_df, x="gender", y="annual_income", hue="membership_status")
    plt.title("Распределение возраста по росту и полу")
    plt.xlabel("Возраст")
    plt.ylabel("Рост")
    plt.xticks(rotation=45)
    plt.legend(title="Не указан пол")
    plt.grid(True)
    plt.show()
    

except Exception as e:
    print("Ошибка выполнения запроса:", e)