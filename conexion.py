import pyodbc

def get_connection():
    try:
        conn_str = (
            "DRIVER={ODBC Driver 17 for SQL Server};"
            "SERVER=localhost;"
            "DATABASE=DHWviolenciaM;"
            "Trusted_Connection=yes;"
        )
        conn = pyodbc.connect(conn_str)
        return conn

    except Exception as e:
        print(f"Error de conexión: {e}")
        return None
