import pandas as pd
from sklearn.impute import SimpleImputer

# Imputación de valores faltantes (Missing Value Imputation):
# Cargar el dataset
data = pd.read_csv('dataset_onehotencoder_scaler.csv')

# Crear un objeto SimpleImputer
imputer = SimpleImputer(strategy='mean')  # Puedes cambiar la estrategia según tus necesidades

# Aplicar la imputación a las características con valores faltantes
data_imputed = imputer.fit_transform(data)

# Crear un nuevo DataFrame con los valores imputados
data_imputed_df = pd.DataFrame(data_imputed, columns=data.columns)

# Guardar el conjunto de datos preprocesado si es necesario
data_imputed_df.to_csv('dataset_imputado.csv', index=False)
