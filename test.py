import pyodbc

# Métodos de conexión a probar
connection_methods = [
    {
        "description": "FORMA ALAN Authentication (usuario CONECTIONBD)",
        "connection_string": (
            "DRIVER={ODBC Driver 17 for SQL Server};"
            "SERVER=localhost;"
            "DATABASE=DHWviolenciaM;"
            "Trusted_Connection=yes;"
        )
    },
    {
        "description": "ALAN2 (usuario CONECTIONBD, usando localhost)",
        "connection_string": (
            "DRIVER={ODBC Driver 17 for SQL Server};"
            "SERVER=localhost\\SQLEXPRESS;"  # Usa localhost como alternativa
            "DATABASE=DWHviolenciaM;"
            "UID=CONECTIONBD;"
            "PWD=123;"
        )
    }
]

# Intentar cada método de conexión y mostrar el resultado
for method in connection_methods:
    try:
        print(f"Intentando {method['description']}...")
        conn = pyodbc.connect(method["connection_string"])
        print(f"Conexión exitosa con {method['description']}")
        conn.close()
        break
    except Exception as e:
        print(f"Error en {method['description']}: {e}")
