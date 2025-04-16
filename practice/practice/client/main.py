import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import base64

st.set_page_config(layout="wide")

API_URL = "http://localhost:8000"

def get_diamonds_data():
    response = requests.get(f"{API_URL}/diamonds")
    print(response)
    print(response.json())
    return response.json()

def get_housing_data():
    response = requests.get(f"{API_URL}/housing")
    print(response)
    print(response.json())
    return response.json()

def show_stats(stats):
    st.subheader("Основные статистики")
    st.write(f"Количество строк: {stats['num_rows']}")
    st.write(f"Количество столбцов: {stats['num_cols']}")
    st.write(f"Используемая память: {stats['memory_usage']:.2f} МБ")
    
    st.subheader("Статистики числовых переменных")
    for col, values in stats['num_stats'].items():
        st.write(f"""
        **{col}**:  
        - Минимум: {values['min']:.2f}  
        - Среднее: {values['mean']:.2f}  
        - Медиана: {values['median']:.2f}  
        - Максимум: {values['max']:.2f}  
        - 25-й перцентиль: {values['25%']:.2f}  
        - 75-й перцентиль: {values['75%']:.2f}
        """)
    
    if stats['cat_stats']:
        st.subheader("Статистики категориальных переменных")
        for col, values in stats['cat_stats'].items():
            st.write(f"""
            **{col}**:  
            - Мода: {values['mode']}  
            - Частота: {values['count']}
            """)

def show_regression(metrics, images):
    st.subheader("Метрики регрессии")
    for model, scores in metrics.items():
        st.write(f"**{model}**")
        st.write(f"MAE: {scores['MAE']:.2f}")
        st.write(f"R²: {scores['R2']:.4f}")
    
    st.subheader("Графики регрессии")
    cols = st.columns(len(images))
    for i, img_base64 in enumerate(images):
        cols[i].image(Image.open(BytesIO(base64.b64decode(img_base64))), use_container_width=True)

def main():
    st.title("Анализ данных")
    
    tab1, tab2 = st.tabs(["KNN", "AutoML"])
    
    with tab1:
        data = get_diamonds_data()
        tab_stats, tab_reg = st.tabs(["Статистика", "Регрессия"])
        
        with tab_stats:
            show_stats(data['stats'])
        
        with tab_reg:
            show_regression(data['regression_metrics'], data['regression_plots'])
    
    with tab2:
        data = get_housing_data()
        tab_stats, tab_reg = st.tabs(["Статистика", "Регрессия"])
        
        with tab_stats:
            show_stats(data['stats'])
        
        with tab_reg:
            show_regression(data['regression_metrics'], data['regression_plots'])

if __name__ == "__main__":
    main()