from sklearn.preprocessing import LabelEncoder, StandardScaler

def transform_data(data):
    le_edad = LabelEncoder()
    le_violencia = LabelEncoder()

    data['Edad_Victima_Coded'] = le_edad.fit_transform(data['Edad_Victima'])
    data['Tipo_Violencia_Coded'] = le_violencia.fit_transform(data['Tipo_Violencia'])

    transformed_data = data[['Edad_Victima_Coded', 'Tipo_Violencia_Coded']]
    return transformed_data, le_edad, le_violencia

def scale_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler
