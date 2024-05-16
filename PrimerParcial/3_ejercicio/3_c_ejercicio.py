import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression

#Selección de características (Feature Selection)
# Cargar el dataset
data = pd.read_csv('dataset_imputado.csv')

# Separar las características y la variable objetivo
X = data.drop(columns=['tension_strenght', 'elongation','roughness'])
y = data['tension_strenght']

# Crear un objeto SelectKBest con el método de puntuación f_regression
selector = SelectKBest(score_func=f_regression, k=5)  # Puedes ajustar el valor de k según tus necesidades

# Aplicar la selección de características
X_selected = selector.fit_transform(X, y)

# Obtener los índices de las características seleccionadas
selected_indices = selector.get_support(indices=True)

# Obtener los nombres de las características seleccionadas
selected_features = X.columns[selected_indices]

# Crear un nuevo DataFrame con las características seleccionadas
data_selected_df = pd.DataFrame(X_selected, columns=selected_features)

# Concatenar la variable objetivo si es necesario
data_selected_df['tension_strenght'] = y

# Guardar el conjunto de datos preprocesado si es necesario
data_selected_df.to_csv('dataset_seleccionado.csv', index=False)
