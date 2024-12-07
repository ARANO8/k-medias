from matplotlib import cm, pyplot as plt

def plot_elbow(inertia):
    import io
    from matplotlib import pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(inertia) + 1), inertia, marker='o')
    plt.title('Método del Codo para Determinar k')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inercia')
    plt.grid(True)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return img


def plot_clusters(transformed_data, scaled_data, kmeans_model):
    import io
    from matplotlib import cm, pyplot as plt

    # Datos para graficar
    colors = cm.get_cmap('viridis', kmeans_model.n_clusters)

    plt.figure(figsize=(10, 6))
    plt.scatter(
        scaled_data[:, 0],  # Edad de la víctima escalada
        scaled_data[:, 1],  # Tipo de violencia escalada
        c=kmeans_model.labels_,
        cmap=colors,
        s=50,
        alpha=0.8,
    )

    # Centroides
    centroids = kmeans_model.cluster_centers_
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=200,
        c="red",
        label="Centroides",
        marker="X",
    )

    plt.title("Visualización de Clusters (k-medias)")
    plt.xlabel("Edad de la Víctima (Escalada)")
    plt.ylabel("Tipo de Violencia (Escalada)")
    plt.colorbar(label="Cluster")
    plt.legend()
    plt.grid(True)

    # Guardar la gráfica en memoria y devolverla
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()  # Cierra la figura para evitar ventanas innecesarias
    return img
