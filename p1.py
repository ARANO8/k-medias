import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from conexion import get_connection

# Función principal
def graficar_edad_vs_tipo_violencia_kmeans():
    # Consulta SQL
    query = """
    SELECT 
        DE.Rango_Edad AS Edad_Victima, 
        TV.Tipo_Violencia, 
        COUNT(HC.ID_Caso) AS Cantidad_Casos
    FROM 
        Hechos_Casos HC
    INNER JOIN 
        Dim_Edad_Victima DE ON HC.ID_Edad_Victima = DE.ID_Edad_Victima
    INNER JOIN 
        Dim_TipoViolencia TV ON HC.ID_Tipo_Violencia = TV.ID_Tipo_Violencia
    GROUP BY 
        DE.Rango_Edad, 
        TV.Tipo_Violencia
    ORDER BY 
        DE.Rango_Edad, 
        TV.Tipo_Violencia;
    """
    
    # Conectar a la base de datos y ejecutar la consulta
    connection = get_connection()
    try:
        df = pd.read_sql_query(query, connection)
    finally:
        connection.close()
    
    # Mostrar los primeros registros (opcional, para verificar)
    print(df.head())
    
    # Codificar variables categóricas a números
    le_edad = LabelEncoder()
    le_tipo = LabelEncoder()
    
    df['Edad_Num'] = le_edad.fit_transform(df['Edad_Victima'])
    df['Tipo_Num'] = le_tipo.fit_transform(df['Tipo_Violencia'])
    
    # Crear un dataset para k-means
    X = df[['Edad_Num', 'Tipo_Num', 'Cantidad_Casos']]
    
    # Aplicar k-means
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    
    # Crear el gráfico con seaborn
    plt.figure(figsize=(12, 8))
    
    # Gráfico de dispersión con clusters
    sns.scatterplot(
        x='Edad_Num', 
        y='Tipo_Num', 
        hue='Cluster', 
        data=df, 
        palette='viridis', 
        size='Cantidad_Casos', 
        sizes=(20, 200), 
        legend='full'
    )
    
    # Añadir los centroides
    plt.scatter(
        centroids[:, 0], 
        centroids[:, 1], 
        s=300, 
        c='red', 
        label='Centroides', 
        marker='X'
    )
    
    # Configurar el gráfico
    plt.title('Clústeres de Edad vs. Tipo de Violencia con K-Means', fontsize=16)
    plt.xlabel('Edad de la Víctima (Codificada)', fontsize=12)
    plt.ylabel('Tipo de Violencia (Codificada)', fontsize=12)
    plt.xticks(ticks=range(len(le_edad.classes_)), labels=le_edad.classes_, rotation=45)
    plt.yticks(ticks=range(len(le_tipo.classes_)), labels=le_tipo.classes_)
    plt.legend()
    plt.tight_layout()
    
    # Mostrar el gráfico
    plt.show()

# Ejecutar la función principal
if __name__ == "__main__":
    graficar_edad_vs_tipo_violencia_kmeans()
