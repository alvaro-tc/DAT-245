import pandas as pd


def calculate_quartile(datos, q):
    """
    Calcula el quantil q para una lista de datos.

    Args:
    - datos: Una lista de números.
    - q: El quantil deseado, debe estar entre 0 y 1.

    Returns:
    - El valor del quantil q.
    """
    datos_ordenados = sorted(datos)
    posQ=(q*(len(datos_ordenados)-1))
    if posQ.is_integer():   
        return datos_ordenados[posQ]
    else:
        vpa=int(posQ)
        vps=vpa+1
        pfp=posQ-vpa
        valQ=(datos_ordenados[vpa]+(datos_ordenados[vps]-datos_ordenados[vpa])*pfp)
        return valQ
    


    
# Cargar el dataset
data = pd.read_csv('data.csv')

# Convertir todas las columnas numéricas a tipo numérico, atrapando excepciones
for column in data.columns:
    try:
        data[column] = pd.to_numeric(data[column])
    except ValueError:
        print(f"La columna {column} no se pudo convertir a tipo numérico.")

# Calcular el último cuartil y percentil 80 por columna
for column in data.select_dtypes(include=['number']).columns:
    last_quartile = calculate_quartile(data[column],0.75)
    percentile_80 = calculate_quartile(data[column],0.8)
    print(f"Columna: {column}, Último Cuartil: {last_quartile}, Percentil 80: {percentile_80}")


