from sklearn.cluster import KMeans

def find_optimal_k(scaled_data, max_k=10):
    inertia = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)
    return inertia

def train_kmeans(scaled_data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_data)
    return kmeans

def predict_violence_type(age, le_edad, scaler, kmeans_model, cluster_summary):
    if age not in le_edad.classes_:
        print(f"Error: La edad '{age}' no está en los datos disponibles.")
        return None

    age_coded = le_edad.transform([age])[0]
    scaled_data = scaler.transform([[age_coded, 0]])
    cluster = kmeans_model.predict(scaled_data)[0]
    return cluster_summary.loc[cluster, 'Tipo_Violencia']
