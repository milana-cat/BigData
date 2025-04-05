import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA, KernelPCA
from scipy.cluster.hierarchy import dendrogram, linkage
import pickle

def load_and_prepare_data(filepath):
    data = pd.read_csv(filepath)
    print("Информация о данных")
    print(data.info())
    print("\nПропущенные значения\n", data.isnull().sum())
    df = preprocess_data(data)
    # Удаление пропущенных значений
    df.dropna(inplace=True)



    return df

def preprocess_data(df):
    """Обрабатывает пропущенные значения и удаляет категориальные признаки."""
    df = df.select_dtypes(include=[np.number])  # Удаляем нечисловые признаки
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    #df_imputed= df_imputed.drop(columns=['car_ID'])
    for colum in df_imputed.columns:
        Q1 = df_imputed[colum].quantile(0.25)
        Q3 = df_imputed[colum].quantile(0.75)
        IQR = Q3 - Q1
        # Удаление выбросов
        filtered_data = df_imputed[~((df_imputed[colum] < (Q1 - 1.5 * IQR)) | (df_imputed[colum] > (Q3 + 1.5 * IQR)))]
    return filtered_data

def scale_features(data, drop_column='Id'):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    return scaled

def elbow_method_manual(X, k_range=range(2, 11)):
    def manual_wss(X, labels, centers):
        wss = 0
        for i in range(centers.shape[0]):
            cluster_points = X[labels == i]
            distances = np.linalg.norm(cluster_points - centers[i], axis=1)
            wss += np.sum(distances ** 2)
        return wss

    wss_values = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
        wss = manual_wss(X, kmeans.labels_, kmeans.cluster_centers_)
        wss_values.append(wss)

    plt.figure()
    plt.plot(k_range, wss_values, marker='o')
    plt.title('Метод локтя (Реализация)')
    plt.xlabel('Число кластеров')
    plt.ylabel('WSS')
    plt.grid(True)
    plt.show()

def silhouette_analysis(X, k_range=range(2, 11)):
    scores = []
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=42).fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append(score)

    plt.figure()
    plt.plot(k_range, scores, marker='o')
    plt.title('Метод силуэта')
    plt.xlabel('Число кластеров')
    plt.ylabel('Силуэтный коэффициент')
    plt.grid(True)
    plt.show()

    return k_range[np.argmax(scores)]

def apply_kmeans(X, k, data):
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X)
    data['kmeans_cluster'] = labels

    plt.figure()
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='viridis')
    plt.title('K-Means Кластеризация')
    plt.show()

    plt.figure()
    unique_labels, counts = np.unique(labels, return_counts=True)
    plt.pie(counts, labels=[f'Cluster {i}' for i in unique_labels], autopct='%1.1ff%%', startangle=140)
    plt.axis('equal')
    plt.title('Cluster Proportions (K-Means)')
    plt.show()

    with open('kmeans_model.pkl', 'wb') as f:
        pickle.dump(model, f)

def apply_hierarchical(X, k, data):
    linkage_matrix = linkage(X, method='ward')
    plt.figure(figsize=(10, 5))
    dendrogram(linkage_matrix)
    plt.title('Дендрограмма')
    plt.xlabel('Образцы')
    plt.ylabel('Евклидово расстояние')
    plt.show()

    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(X)
    data['hierarchical_cluster'] = labels

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=labels, palette='Set2')
    plt.title('Иерархическая кластеризация в 2D (PCA)')
    plt.xlabel('PCA компонент 1')
    plt.ylabel('PCA компонент 2')
    plt.grid(True)
    plt.legend(title='Кластер')
    plt.show()

    plt.figure()
    unique_labels, counts = np.unique(labels, return_counts=True)
    plt.pie(counts, labels=[f'Cluster {i}' for i in unique_labels], autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Cluster Proportions (Hierarchical)')
    plt.show()

    with open('hierarchical_model.pkl', 'wb') as f:
        pickle.dump(model, f)


def custom_kmeans(X, n_clusters, max_iters=100, random_state=42):
    np.random.seed(random_state)
    # 1. Случайный выбор центров
    indices = np.random.choice(len(X), n_clusters, replace=False)
    centers = X[indices]

    for iteration in range(max_iters):
        # 2. Присвоение точек к ближайшему центру
        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        labels = np.argmin(distances, axis=1)

        # 3. Перерасчет центров
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])

        # 4. Проверка сходимости
        if np.allclose(centers, new_centers):
            break
        centers = new_centers

    return labels, centers


def main():
    data = load_and_prepare_data('WineQT.csv')
    kpca = KernelPCA(n_components=3, kernel ='cosine');
    X_scaled = kpca.fit_transform(data)
    #X_scaled = scale_features(X_scaled )


    elbow_method_manual(X_scaled)
    optimal_k = silhouette_analysis(X_scaled)
    print(f'Оптимальное число кластеров: {optimal_k}')


       # ❶ Самостоятельная реализация K-Means
    custom_labels, custom_centers = custom_kmeans(X_scaled, optimal_k)
    data['custom_kmeans'] = custom_labels
    
   
    

    apply_kmeans(X_scaled, optimal_k, data)
    

     # ❷ Визуализация для сравнения
    from sklearn.decomposition import PCA
    data_pca = PCA(n_components=2).fit_transform(X_scaled)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=custom_labels, palette='Set1')
    plt.title("Custom K-Means")


    plt.subplot(1, 2, 2)
    sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=data['kmeans_cluster'], palette='Set2')
    plt.title("Sklearn K-Means")
    plt.tight_layout()
    plt.show()

    apply_hierarchical(X_scaled, optimal_k, data)
    print("Кластеризация завершена!")

if __name__ == "__main__":
    main()
