from sqlalchemy import create_engine
import pandas as pd
from scipy.stats import f_oneway, ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt

def connect_to_db(host, port, user, password, database):
    #Подключение к базе данных PostgreSQL."""
    try:
        engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{database}?client_encoding=utf8")
        print("Подключение успешно")
        return engine
    except Exception as e:
        print("Ошибка подключения:", e)
        exit()

def load_data(engine, query):
   #Загрузка данных из базы данных в DataFrame.
    try:
        df = pd.read_sql_query(query, engine)
        print("Данные успешно загружены в DataFrame")
        return df
    except Exception as e:
        print("Ошибка выполнения запроса:", e)
        exit()

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
            print(f"Дисперсия: {df[col].var():.4f}")
            print(f"Квантиль 0.1: {df[col].quantile(0.1):.4f}")
            print(f"Квантиль 0.9: {df[col].quantile(0.9):.4f}")
            print(f"Квартиль 1: {df[col].quantile(0.25):.4f}")
            print(f"Квартиль 3: {df[col].quantile(0.75):.4f}")
        except Exception as e:
            print(f"Ошибка анализа переменной {col}: {e}")

def exploratory_analysis_categorical(df, columns):
    #Разведочный анализ категориальных переменных.
    for col in columns:
        print(f"\nАнализ категориальной переменной: {col}")
        print(f"Доля пропусков: {df[col].isnull().mean():.4%}")
        print(f"Количество уникальных значений: {df[col].nunique()}")
        print(f"Мода: {df[col].mode()[0] if not df[col].mode().empty else 'Нет моды'}")

def encode_categorical_variables(df):
    #Пошаговое кодирование категориальных переменных.
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_columns:
        if col !='name' and col !='id':
            df[col], unique_values = pd.factorize(df[col])
            print(f"\nСтолбец '{col}' закодирован. Первые 5 значений после кодирования:")
            print(df[col].head())
            print(f"Уникальные категории: {list(unique_values[:5])} ...")
    return df

def analyze_dataset(df):
    #Подсчёт строк и столбцов.
    rows, columns = df.shape
    print(f"Количество строк: {rows}, Количество столбцов: {columns}")

def test_hypotheses(df):
    #Проверка статистических гипотез.
    print("\nГипотеза 1: Различается ли доход за год (annual_income) для с статусом членства (membership_status)?")
    silver_weight = df[df['membership_status'] == 0]['annual_income'].dropna()
    gold_weight = df[df['membership_status'] == 1]['annual_income'].dropna()
    stat, p_value = ttest_ind(gold_weight , silver_weight, equal_var=False)
    print(f"t-test: статистика = {stat:.2f}, p-значение = {p_value:.2e}")
    if p_value < 0.05:
        print("Отвергаем гипотезу: доход за год различается у людей с разным статусом членства.")
    else:
        print("Не удалось отвергнуть гипотезу: доход за год не различается у людей с разным статусом членства.")

    print("\nГипотеза 2: Различается ли доход за год (annual_income) у мужчин и женщин ?")
    man_weight = df[df['gender'] == 0]['annual_income'].dropna()
    woman_weight = df[df['gender'] == 1]['annual_income'].dropna()
    stat, p_value = ttest_ind(man_weight , woman_weight , equal_var=False)
    print(f"t-test: статистика = {stat:.2f}, p-значение = {p_value:.2e}")
    if p_value < 0.05:
        print("Отвергаем гипотезу: доход за год различается у мужчин и женщин.")
    else:
        print("Не удалось отвергнуть гипотезу: доход за год не различается у мужчин и женщин.")

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

if __name__ == "__main__":
    # Данные для подключения
    host = "povt-cluster.tstu.tver.ru"
    port = 5432
    user = "mpi"
    password = "135a1"
    database = "Criminal"
    
    # Подключение к базе данных
    engine = connect_to_db(host, port, user, password, database)
    
    # Загрузка данных
    query = 'SELECT * FROM person JOIN get_fit_now_member ON person.id = get_fit_now_member.person_id JOIN drivers_license ON person.license_id = drivers_license.id JOIN income ON income.ssn = person.ssn  LIMIT 1000'
    df = load_data(engine, query)
    
    # Определение признаков
    nominal_columns = [   "address_street_name",  "membership_status", "eye_color", "hair_color", "gender", "car_make", "car_model"]
    ordinal_columns = ["membership_start_date"]
    quantitative_columns = ["age", "height","annual_income"]
    exploratory_analysis_categorical(df,nominal_columns )
    # Кодирование категориальных переменных
    df = encode_categorical_variables(df)
    
    # Анализ набора данных
    analyze_dataset(df)
    exploratory_analysis_numeric(df, quantitative_columns)
  
    # Проверка статистических гипотез
    test_hypotheses(df)
    
    # Построение таблицы корреляции
    correlation_table(df, target_column="membership_status")
