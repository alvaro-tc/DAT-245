import pandas as pd
from sklearn.preprocessing import StandardScaler
#Normalización de características (Feature Normalization)
#para la media 0 y la desviacion estandar 1
# Cargar el dataset
data = pd.read_csv('dataset_seleccionado.csv')

# Separar las características y la variable objetivo si es necesario
X = data.drop( columns=['tension_strenght'])
y = data['tension_strenght']

# Crear un objeto StandardScaler
scaler = StandardScaler()

# Aplicar la normalización a las características
X_normalized = scaler.fit_transform(X)

# Crear un nuevo DataFrame con las características normalizadas
data_normalized_df = pd.DataFrame(X_normalized, columns=X.columns)

# Concatenar la variable objetivo si es necesario
data_normalized_df['tension_strenght'] = y

# Guardar el conjunto de datos preprocesado si es necesario
data_normalized_df.to_csv('dataset_normalizado.csv', index=False)
