import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA, PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE

def load_data(filepath):
    """Загружает датасет из CSV файла."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Обрабатывает пропущенные значения и удаляет категориальные признаки."""
    df = df.select_dtypes(include=[np.number])  # Удаляем нечисловые признаки
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    df_imputed= df_imputed.drop(columns=['Id'])
    for colum in df_imputed.columns:
        Q1 = df_imputed[colum].quantile(0.25)
        Q3 = df_imputed[colum].quantile(0.75)
        IQR = Q3 - Q1
        # Удаление выбросов
        filtered_data = df_imputed[~((df_imputed[colum] < (Q1 - 1.5 * IQR)) | (df_imputed[colum] > (Q3 + 1.5 * IQR)))]
    return filtered_data

def eda_analysis(df):
    """Выполняет разведочный анализ данных (EDA)."""
    print("\nОсновная информация о данных \n")
    print(df.info())
    print("\nПервые 5 строк данных \n")
    print(df.head())
    print("\nОписательная статистика\n")
    print(df.describe())
    
    # Проверка пропущенных значений
    print("\nПропущенные значения\n")
    print(df.isnull().sum())
    
    # Построение гистограмм для каждого признака
    df.hist(figsize=(12, 10), bins=20)
    plt.show()
    
    # Корреляционная матрица
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Корреляционная матрица")
    plt.show()
    
    return df

def normalize_data(df):
    """Нормализация данных с помощью StandardScaler."""
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled

def apply_kernel_pca(df, target):
    """Применяет Kernel PCA с разными ядрами и окрашивает точки по значениям target."""
    kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']
    kpca_results = {}
    
    for kernel in kernels:
        kpca = KernelPCA(n_components=3, kernel=kernel)
        df_kpca = kpca.fit_transform(df)
        kpca_results[kernel] = df_kpca
        
        plt.figure()
        scatter = plt.scatter(df_kpca[:, 0], df_kpca[:, 1], c=target, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Target Value')
        plt.title(f"Kernel PCA с ядром: {kernel}")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.show()
    
    return kpca_results

def apply_tsne(df, target):
    """Применяет t-SNE для снижения размерности и окрашивает точки по значениям target."""
    tsne = TSNE(n_components=3, perplexity=50, random_state=42)
    df_tsne = tsne.fit_transform(df)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(df_tsne[:, 0], df_tsne[:, 1], c=target, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Target Value')
    plt.title("t-SNE визуализация")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()
    
   

def analyze_linear_pca(df, target):
    """Анализ дисперсии и вычисление потери информации для линейного ядра с цветовой визуализацией."""
    pca_full = PCA()
    pca_full.fit(df)
    total_variance = np.sum(pca_full.explained_variance_ratio_)
    
    pca_reduced = KernelPCA(n_components=10, kernel='linear', fit_inverse_transform=True)
    df_reduced = pca_reduced.fit_transform(df)
    df_reconstructed = pca_reduced.inverse_transform(df_reduced)
    
    pca_after = PCA()
    pca_after.fit(df_reconstructed)
    reduced_variance = np.sum(pca_after.explained_variance_ratio_)
    lost_variance = total_variance - reduced_variance
    
    print(f"Общая дисперсия исходных данных: {total_variance:.4f}")
    print(f"Общая дисперсия после Kernel PCA (linear): {reduced_variance:.4f}")
    print(f"Потеря дисперсии (lost_variance): {lost_variance:.4f}")
    
    plt.figure()
    scatter = plt.scatter(df_reduced[:, 0], df_reduced[:, 1], c=target, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Target Value')
    plt.title("Kernel PCA (Linear) Visualization")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()



def main():
    filepath = "WineQT.csv"  #Исключить квалити и выделить цветом. 
    df = load_data(filepath)
    df = preprocess_data(df)  #Обработка пропущенных значений и удаление категориальных признаков
    df = eda_analysis(df)
    df = normalize_data(df)
    print("\nДанные после нормализации:\n", df.head())
    
    # Применение Kernel PCA
    apply_kernel_pca(df.drop(columns=['quality']), df['quality'])
    
    # Применение t-SNE
    apply_tsne(df.drop(columns=['quality']), df['quality'])
    
    # Анализ линейного PCA
    analyze_linear_pca(df.drop(columns=['quality']), df['quality'])

if __name__ == "__main__":
    main()
