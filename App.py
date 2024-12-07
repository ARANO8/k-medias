from flask import Flask, render_template, request
import pandas as pd
from clustering import train_kmeans, predict_violence_type, find_optimal_k
from data_processing import transform_data, scale_data
from visualization import plot_elbow, plot_clusters
from conexion import get_engine
import base64

app = Flask(__name__)

# Cargar y procesar datos
engine = get_engine()
query = """
SELECT dv.Rango_Edad AS Edad_Victima, tv.Tipo_Violencia
FROM Hechos_Casos hc
JOIN Dim_Edad_Victima dv ON hc.ID_Edad_Victima = dv.ID_Edad_Victima
JOIN Dim_TipoViolencia tv ON hc.ID_Tipo_Violencia = tv.ID_Tipo_Violencia
"""
data = pd.read_sql(query, engine)
transformed_data, le_edad, le_violencia = transform_data(data)
scaled_data, scaler = scale_data(transformed_data)

# Entrenar modelo
k = 4
kmeans_model = train_kmeans(scaled_data, n_clusters=k)
transformed_data.loc[:, 'Cluster'] = kmeans_model.labels_

# Resumen de clusters
cluster_summary = pd.DataFrame({
    'Cluster': transformed_data['Cluster'],
    'Tipo_Violencia': data['Tipo_Violencia']
}).groupby('Cluster').agg(lambda x: x.mode()[0])

@app.route('/')
def index():
    # Gráfica del método del codo
    inertia = find_optimal_k(scaled_data)
    codo_img = plot_elbow(inertia)

    # Gráfica de clusters
    cluster_img = plot_clusters(transformed_data, scaled_data, kmeans_model)

    # Convertir imágenes a Base64
    codo_data = base64.b64encode(codo_img.getvalue()).decode()
    cluster_data = base64.b64encode(cluster_img.getvalue()).decode()

    return render_template('index.html', codo_data=codo_data, cluster_data=cluster_data, cluster_summary=cluster_summary.to_html())

@app.route('/predict', methods=['POST'])
def predict():
    age = request.form['age']
    predicted_type = predict_violence_type(age, le_edad, scaler, kmeans_model, cluster_summary)
    return render_template('predict.html', age=age, predicted_type=predicted_type)

if __name__ == '__main__':
    app.run(debug=True)
