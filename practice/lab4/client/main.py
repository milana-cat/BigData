import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import base64

st.set_page_config(layout="wide")

API_URL = "http://localhost:8501"

# Безопасный запрос данных
def get_data(endpoint):
    try:
        response = requests.get(f"{API_URL}/{endpoint}", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Ошибка при подключении к API ({endpoint}): {e}")
        return None

def show_stats(stats):
    if not stats:
        st.warning("Нет статистических данных для отображения.")
        return

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
    if not metrics:
        st.warning("Метрики регрессии отсутствуют.")
        return

    st.subheader("Метрики регрессии")
    for model, scores in metrics.items():
        st.write(f"**{model}**")
        st.write(f"MAE: {scores['MAE']:.2f}")
        st.write(f"R²: {scores['R2']:.4f}")
    
    if images:
        st.subheader("Графики регрессии")
        cols = st.columns(len(images))
        for i, img_base64 in enumerate(images):
            try:
                image = Image.open(BytesIO(base64.b64decode(img_base64)))
                cols[i].image(image, use_container_width=True)
            except Exception as e:
                st.error(f"Ошибка отображения изображения: {e}")
    else:
        st.info("Графики отсутствуют.")

def main():
    st.title("📊 Анализ табличных данных")
    tab1, tab2 = st.tabs(["AutoML", "🍷 WineQT"])
    
    with tab1:
        data = get_data("AutoML")
        if data:
            tab_stats, tab_reg = st.tabs(["Статистика", "Регрессия"])
            with tab_stats:
                show_stats(data.get('stats'))
            with tab_reg:
                show_regression(data.get('regression_metrics'), data.get('regression_plots'))

    with tab2:
        data = get_data("housing")
        if data:
            tab_stats, tab_reg = st.tabs(["Статистика", "Регрессия"])
            with tab_stats:
                show_stats(data.get('stats'))
            with tab_reg:
                show_regression(data.get('regression_metrics'), data.get('regression_plots'))

if __name__ == "__main__":
    main()
