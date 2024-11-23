import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from conexion import get_connection

# 1. Conexión y extracción de datos
def fetch_data():
    conn = get_connection()
    if conn:
        query = """
        SELECT 
            hc.ID_Caso, 
            dl.Ciudad, 
            dl.Departamento, 
            dt.Tipo_Violencia, 
            dev.Rango_Edad AS Edad_Victima, 
            dea.Rango_Edad AS Edad_Agresor, 
            dr.Relacion, 
            dres.Respuesta
        FROM Hechos_Casos hc
        JOIN Dim_Localizacion dl ON hc.ID_Localizacion = dl.ID_Localizacion
        JOIN Dim_TipoViolencia dt ON hc.ID_Tipo_Violencia = dt.ID_Tipo_Violencia
        JOIN Dim_Edad_Victima dev ON hc.ID_Edad_Victima = dev.ID_Edad_Victima
        JOIN Dim_Edad_Agresor dea ON hc.ID_Edad_Agresor = dea.ID_Edad_Agresor
        JOIN Dim_Relacion dr ON hc.ID_Relacion = dr.ID_Relacion
        JOIN Dim_Respuesta dres ON hc.ID_Respuesta = dres.ID_Respuesta
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

# 3. Aplicar K-Medias
def apply_kmeans(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    return kmeans, clusters

# 4. Visualización
def plot_clusters(data, clusters):
    plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    plt.title('Clusters de Casos')
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.colorbar()
    plt.show()

# Main
if __name__ == "__main__":
    df = fetch_data()
    if df is not None:
        print("Datos cargados con éxito.")
        print(df.head())

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

        # Aplicar K-Medias con un número específico de clusters
        kmeans, clusters = apply_kmeans(processed_data, n_clusters=3)
        df['Cluster'] = clusters
        print(df[['ID_Caso', 'Cluster']].head())

        # Visualización de los clusters
        plot_clusters(processed_data, clusters)
    else:
        print("Error al cargar los datos.")
