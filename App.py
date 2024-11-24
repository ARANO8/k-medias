# Importación de bibliotecas necesarias
import pandas as pd  # Para trabajar con estructuras de datos como DataFrames
from sklearn.preprocessing import MinMaxScaler, LabelEncoder  # Para escalar y convertir datos categóricos en números
from sklearn.cluster import KMeans  # Para aplicar el algoritmo K-Medias de clustering
import matplotlib.pyplot as plt  # Para crear gráficos y visualizaciones
from conexion import get_connection  # Para establecer la conexión con la base de datos

# 1. Conexión y extracción de datos
def fetch_data():
    # Conexión a la base de datos usando una función externa (get_connection)
    conn = get_connection()
    if conn:  # Si la conexión es exitosa
        # Consulta SQL para obtener los datos de la base de datos
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
        # Ejecución de la consulta y carga de los datos en un DataFrame
        df = pd.read_sql(query, conn)
        conn.close()  # Cierre de la conexión a la base de datos
        return df  # Retorna el DataFrame con los datos obtenidos
    else:
        # Si no se pudo establecer la conexión, se retorna None
        return None

# 2. Preprocesamiento de datos
def preprocess_data(df):
    # Convertir las variables categóricas a valores numéricos para su análisis
    label_encoders = {}  # Diccionario para almacenar los codificadores de cada columna
    for column in ['Ciudad', 'Departamento', 'Tipo_Violencia', 'Edad_Victima', 'Edad_Agresor', 'Relacion', 'Respuesta']:
        le = LabelEncoder()  # Instanciamos un codificador para la columna
        df[column] = le.fit_transform(df[column])  # Codificamos la columna y reemplazamos en el DataFrame
        label_encoders[column] = le  # Guardamos el codificador para poder revertir la codificación si es necesario

    # Escalado de los datos para normalizarlos en un rango de 0 a 1
    scaler = MinMaxScaler()  # Instanciamos un escalador que convierte los valores entre 0 y 1
    # Aplicamos el escalado a todas las columnas excepto 'ID_Caso' (que no es necesaria para el análisis)
    scaled_data = scaler.fit_transform(df.drop(columns=['ID_Caso']))
    # print(label_encoders)
    # print(scaled_data)
    return scaled_data, label_encoders  # Retornamos los datos escalados y los codificadores

# 3. Aplicar K-Medias
def apply_kmeans(data, n_clusters=3):
    # Creamos una instancia del algoritmo KMeans con el número de clusters especificado
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    # Aplicamos el algoritmo KMeans al conjunto de datos para asignar cada punto a su cluster más cercano
    clusters = kmeans.fit_predict(data)
    return kmeans, clusters  # Retornamos el modelo KMeans entrenado y las etiquetas de clusters

# 4. Visualización de los resultados de los clusters
def plot_clusters(data, clusters, centroids):
    plt.figure(figsize=(12, 8))  # Definimos el tamaño del gráfico

    # Gráfico de dispersión de los puntos de datos, coloreados por el cluster al que pertenecen
    
    plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='plasma', alpha=0.6)
    
    # Graficamos los centroides de cada cluster como puntos rojos
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X')

    # Añadimos el título y las etiquetas de los ejes
    plt.title('Clusters de Casos: Edad de la Víctima vs. Tipo de Violencia (Escalado)', fontsize=16)
    plt.xlabel('Edad de la Víctima (Escalado)', fontsize=12)
    plt.ylabel('Tipo de Violencia (Escalado)', fontsize=12)
    plt.legend()  # Añadimos una leyenda al gráfico
    # plt.grid(True)  # Puedes descomentar esta línea si deseas agregar una cuadrícula al gráfico
    plt.show()  # Mostramos el gráfico

# Main: ejecuta el proceso completo
if __name__ == "__main__":
    # Llamamos a la función fetch_data() para obtener los datos
    df = fetch_data()
    if df is not None:  # Si los datos se cargaron correctamente
        print("Datos cargados con éxito.")  # Imprimimos un mensaje de éxito
        print(df.head())  # Mostramos las primeras filas del DataFrame para inspección

        # Preprocesamos los datos (escalado y codificación de variables)
        processed_data, encoders = preprocess_data(df)

        # Determinar el número óptimo de clusters (opcional)
        distortions = []  # Lista para almacenar la inercia de cada número de clusters
        for k in range(1, 10):  # Probar con diferentes valores de k (número de clusters)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # Creamos el modelo KMeans
            kmeans.fit(processed_data)  # Entrenamos el modelo con los datos procesados
            distortions.append(kmeans.inertia_)  # Guardamos la inercia para el gráfico del codo
        # Graficamos el método del codo para encontrar el número óptimo de clusters
        plt.plot(range(1, 10), distortions, marker='o')
        plt.title('Método del codo')  # Título del gráfico
        plt.xlabel('Número de clusters')  # Etiqueta del eje X
        plt.ylabel('Inercia')  # Etiqueta del eje Y
        plt.show()  # Mostramos el gráfico

        # Aplicamos K-Medias con 3 clusters
        kmeans, clusters = apply_kmeans(processed_data, n_clusters=3)
        df['Cluster'] = clusters  # Añadimos la etiqueta de cluster al DataFrame
        print(df[['ID_Caso', 'Cluster']].head())  # Mostramos los primeros casos con sus clusters
        
        print("Datos:")
        print(processed_data)
        
        # Imprimimos las coordenadas de los centroides        
        print("Los centroides")
        print(kmeans.cluster_centers_)

        # Visualizamos los clusters
        plot_clusters(processed_data, clusters, kmeans.cluster_centers_)
    else:
        # Si hubo un error al cargar los datos, imprimimos un mensaje
        print("Error al cargar los datos.")
