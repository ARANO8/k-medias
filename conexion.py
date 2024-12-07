from sqlalchemy import create_engine

def get_engine():
    try:
        engine = create_engine(
            "mssql+pyodbc://localhost/DHWviolenciaM?driver=ODBC+Driver+17+for+SQL+Server"
        )
        return engine
    except Exception as e:
        print(f"Error al crear el motor de conexión: {e}")
        return None
