import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Cargar el dataset
data = pd.read_csv('data.csv')

# Seleccionar las columnas que son variables nominales
categorical_columns = ['infill_pattern', 'material']

# Crear un objeto OneHotEncoder
onehot_encoder = OneHotEncoder()

# Aplicar OneHotEncoder a las variables nominales y convertirlas en columnas binarias
onehot_encoded = onehot_encoder.fit_transform(data[categorical_columns])

# Crear nombres para las nuevas columnas generadas
onehot_encoded_columns = onehot_encoder.get_feature_names_out(categorical_columns)

# Crear un DataFrame con las columnas binarias generadas
onehot_encoded_df = pd.DataFrame(onehot_encoded.toarray(), columns=onehot_encoded_columns)

# Eliminar las columnas categóricas originales del DataFrame
data_numeric = data.drop(columns=categorical_columns)

# Escalar las características numéricas
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_numeric)

# Crear un DataFrame con las características numéricas escaladas
data_scaled_df = pd.DataFrame(data_scaled, columns=data_numeric.columns)

# Concatenar las nuevas columnas binarias y las características numéricas escaladas al DataFrame original
data_encoded_scaled = pd.concat([data_scaled_df, onehot_encoded_df], axis=1)

data_encoded_scaled.to_csv('dataset_onehotencoder_scaler.csv', index=False)
