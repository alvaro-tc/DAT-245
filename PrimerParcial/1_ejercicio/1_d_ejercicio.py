import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el conjunto de datos
data = pd.read_csv('data.csv')

# Configurar el estilo de gráficos
sns.set(style="ticks")

# Crear un gráfico de dispersión entre 'layer_height' y 'roughness'
plt.figure(figsize=(8, 6))
sns.scatterplot(x='layer_height', y='roughness', data=data)
plt.title('Relación entre Layer Height y Roughness')
plt.xlabel('Layer Height')
plt.ylabel('Roughness')
plt.show()



# Seleccionar las variables de interés
infill_density = data['infill_density']
wall_thickness = data['wall_thickness']
tension_strenght = data['tension_strenght']

# Configurar la figura y el eje 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Crear el gráfico 3D
ax.scatter(infill_density, wall_thickness, tension_strenght, c='b', marker='o')

# Etiquetas de los ejes
ax.set_xlabel('Infill Density')
ax.set_ylabel('Wall Thickness')
ax.set_zlabel('Tension Strenght')

# Título del gráfico
plt.title('Relación entre Infill Density, Wall Thickness y Tension Strenght (3D)')

# Mostrar el gráfico
plt.show()