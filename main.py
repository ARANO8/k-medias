import pandas as pd
from conexion import get_engine
from data_processing import transform_data, scale_data
from clustering import find_optimal_k, train_kmeans, predict_violence_type
from visualization import plot_elbow, plot_clusters

if __name__ == "__main__":
    # Obtener datos
    engine = get_engine()
    if engine:
        query = """
        SELECT dv.Rango_Edad AS Edad_Victima, tv.Tipo_Violencia
        FROM Hechos_Casos hc
        JOIN Dim_Edad_Victima dv ON hc.ID_Edad_Victima = dv.ID_Edad_Victima
        JOIN Dim_TipoViolencia tv ON hc.ID_Tipo_Violencia = tv.ID_Tipo_Violencia
        """
        data = pd.read_sql(query, engine)

        # Transformar y escalar datos
        transformed_data, le_edad, le_violencia = transform_data(data)
        scaled_data, scaler = scale_data(transformed_data)

        # Determinar el número de clusters
        inertia = find_optimal_k(scaled_data)
        plot_elbow(inertia)

        # Entrenar el modelo
        k = 4
        kmeans_model = train_kmeans(scaled_data, n_clusters=k)
        transformed_data.loc[:, 'Cluster'] = kmeans_model.labels_

        # Resumen de clusters
        cluster_summary = pd.DataFrame({
            'Cluster': transformed_data['Cluster'],
            'Tipo_Violencia': data['Tipo_Violencia']
        }).groupby('Cluster').agg(lambda x: x.mode()[0])

        print("Resumen de Clusters y Tipos de Violencia:")
        print(cluster_summary)

        # Visualizar clusters
        plot_clusters(transformed_data, scaled_data, kmeans_model)

        # Predicción
        age = input("Ingrese la edad de la víctima (en formato de rango como '18-23 años'): ")
        predicted_type = predict_violence_type(age, le_edad, scaler, kmeans_model, cluster_summary)
        if predicted_type:
            print(f"Tipo de violencia predicho para la edad ingresada: {predicted_type}")
    else:
        print("No se pudo conectar a la base de datos.")
