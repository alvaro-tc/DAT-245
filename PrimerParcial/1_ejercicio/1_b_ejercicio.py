import numpy as np
import pandas as pd
import math

data = pd.read_csv('data.csv')


for column in data.columns:
    try:
        data[column] = pd.to_numeric(data[column])
    except ValueError:
        print(f"La columna {column} no se pudo convertir a tipo numérico.")

# Calcular el último cuartil y percentil 80 por columna
for column in data.select_dtypes(include=['number']).columns:
    last_quartile = data[column].quantile(0.75,interpolation='linear')
    percentile_80 = data[column].quantile(0.8,interpolation='linear')
    print(f"Columna: {column}, Último Cuartil: {last_quartile}, Percentil 80: {percentile_80}")
    last_quartile = np.quantile(data[column],0.75)
    percentile_80 = np.quantile(data[column],0.8)
    print(f"Columna: {column}, Último Cuartil: {last_quartile}, Percentil 80: {percentile_80}")
    