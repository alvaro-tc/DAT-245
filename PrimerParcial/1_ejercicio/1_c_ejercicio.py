import numpy as np
import pandas as pd
import math

data = pd.read_csv('data.csv')

def calcular_moda(datos):
    valores, conteos = np.unique(datos, return_counts=True)
    indice_moda = np.argmax(conteos)
    moda = valores[indice_moda]
    return moda

for column in data.columns:
    try:
        data[column] = pd.to_numeric(data[column])
    except ValueError:
        print(f"La columna {column} no se pudo convertir a tipo numérico.")

for column in data.select_dtypes(include=['number']).columns:
    media = np.mean(data[column])
    mediana = np.median(data[column])
    moda = calcular_moda(data[column])
    media_geom = np.prod(data[column]) ** (1 / len(data[column]))
    print(f"Columna: {column}, media: {media}, mediana: {mediana}, moda: {moda}, media geometrica: {media_geom}")