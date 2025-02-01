from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick



def main():
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

    make_plot_single(filtered_df)
     # Многомерный анализ: Построение графиков
    make_plot_multy(filtered_df)
    make_boxplot(filtered_df)  

 except Exception as e:
    print("Ошибка выполнения запроса:", e)

# Одномерный анализ: Построение гистограмм
def make_plot_single(data):
    for feature in ["age", "membership_status"]:  # Количественные признаки
        if feature in data.columns:
            plt.figure(figsize=(8, 5))
            plt.hist(data[feature].dropna(), bins=30, edgecolor="k", alpha=0.7)
            plt.title(f"Распределение признака: {feature}")
            plt.xlabel("Статус членства" if feature == "membership_status" else feature)
            plt.ylabel("Частота")

            plt.grid(True)
            plt.show()
        else:
            print(f"Признак '{feature}' не найден в данных.")

def make_plot_multy(data):
        # График 1: Зависимость saleprice от acreage без категорий SINGLE FAMILY
    filtered_data_no_single_family = data[data["membership_status"] != "regular"]

    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        data=filtered_data_no_single_family,
        x="annual_income",
        y="age",
        hue="gender",
        alpha=0.7
    )
    plt.title("Зависимость годового дохода от возраста без членства regular")
    plt.xlabel("заработок за год")
    plt.ylabel("возраст")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))  # Форматирование оси Y
    plt.grid(True)
    plt.legend(title="Зависимость годового дохода от возраста")
    plt.show()

def make_boxplot(data):
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data, x="gender", y="annual_income", hue="membership_status")
    plt.title("Распределение доходов по статусу членства и полу")
    plt.xlabel("пол")
    plt.ylabel("годовой доход")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

main()