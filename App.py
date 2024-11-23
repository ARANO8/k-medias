import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from conexion import get_connection

# 1. Conexión y extracción de datos
def fetch_data():
    conn = get_connection()
    if conn:
        query = """
        SELECT 
            ID_Tipo_Violencia, 
            ID_Edad_Victima
        FROM Hechos_Casos
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    else:
        return None
# 2. Preprocesamiento de datos
def preprocess_data(df):
    # Convertir variables categóricas a numéricas
    label_encoders = {}
    for column in ['Ciudad', 'Departamento', 'Tipo_Violencia', 'Edad_Victima', 'Edad_Agresor', 'Relacion', 'Respuesta']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Escalado de los datos
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.drop(columns=['ID_Caso']))
    return scaled_data, label_encoders
# 2. Aplicar K-Medias
def apply_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    return kmeans, clusters

# 3. Visualización de clusters
def plot_clusters(data, clusters, kmeans):
    plt.figure(figsize=(8, 6), dpi=100)
    colores = ["red", "blue", "orange", "black", "purple", "pink", "brown"]
    
    # Graficar los puntos según el cluster
    for cluster in range(kmeans.n_clusters):
        plt.scatter(data[clusters == cluster, 0], 
                    data[clusters == cluster, 1],
                    marker="o", s=180, color=colores[cluster], alpha=0.5)
        
        # Graficar los centroides
        plt.scatter(kmeans.cluster_centers_[cluster][0], 
                    kmeans.cluster_centers_[cluster][1], 
                    marker="P", s=280, color=colores[cluster])

    plt.title("Clusters de Casos", fontsize=20)
    plt.xlabel("ID_Tipo_Violencia (Eje X)", fontsize=15)
    plt.ylabel("ID_Edad_Victima (Eje Y)", fontsize=15)
    plt.text(max(data[:, 0]) + 1, max(data[:, 1]) - 1, f"K = {kmeans.n_clusters}", fontsize=15)
    plt.text(max(data[:, 0]) + 1, max(data[:, 1]) - 3, f"Inercia = {kmeans.inertia_:.2f}", fontsize=15)
    plt.show()

# Main
if __name__ == "__main__":
    df = fetch_data()
    if df is not None:
        print("Datos cargados con éxito.")
        print(df.head())

        # Convertir el DataFrame a numpy para el modelo
        data = df.values
        # Preprocesamiento
        processed_data, encoders = preprocess_data(df)
        # Determinar el número óptimo de clusters (opcional)
        # Método del codo
        distortions = []
        for k in range(1, 10):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(processed_data)
            distortions.append(kmeans.inertia_)
        plt.plot(range(1, 10), distortions, marker='o')
        plt.title('Método del codo')
        plt.xlabel('Número de clusters')
        plt.ylabel('Inercia')
        plt.show()
        
        # Pedir el número de clusters al usuario
        try:
            k = int(input("Ingrese el número de clusters (k): "))
            if k < 1:
                raise ValueError("El número de clusters debe ser al menos 1.")
        except ValueError as e:
            print(f"Error: {e}. Se usará k=3 por defecto.")
            k = 3

        # Aplicar K-Medias
        kmeans, clusters = apply_kmeans(data, n_clusters=k)

        # Visualizar los clusters
        plot_clusters(data, clusters, kmeans)
    else:
        print("Error al cargar los datos.")
