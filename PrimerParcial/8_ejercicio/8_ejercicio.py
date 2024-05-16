import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

distance_matrix=np.array([
    [0, 10, 15, 20, 25, 30, 35, 40],
    [10, 0, 12, 18, 21, 28, 32, 36],
    [15, 12, 0, 8, 16, 24, 29, 33],
    [20, 18, 8, 0, 5, 12, 17, 21],
    [25, 21, 16, 5, 0, 6, 11, 15],
    [30, 28, 24, 12, 6, 0, 6, 10],
    [35, 32, 29, 17, 11, 6, 0, 5],
    [40, 36, 33, 21, 15, 10, 5, 0]
])


# Definir el grafo
G = nx.Graph()

# Agregar nodos al grafo
num_ciudades = 8
nodos = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
G.add_nodes_from(nodos)

# Agregar aristas ponderadas al grafo
aristas = [
    ('A', 'B', 10),
    ('A', 'C', 15),
    ('A', 'D', 20),
    ('A', 'E', 25),
    ('A', 'F', 30),
    ('A', 'G', 35),
    ('A', 'H', 40),
    ('B', 'C', 12),
    ('B', 'D', 18),
    ('B', 'E', 21),
    ('B', 'F', 28),
    ('B', 'G', 32),
    ('B', 'H', 36),
    ('C', 'D', 8),
    ('C', 'E', 16),
    ('C', 'F', 24),
    ('C', 'G', 29),
    ('C', 'H', 33),
    ('D', 'E', 5),
    ('D', 'F', 12),
    ('D', 'G', 17),
    ('D', 'H', 21),
    ('E', 'F', 6),
    ('E', 'G', 11),
    ('E', 'H', 15),
    ('F', 'G', 6),
    ('F', 'H', 10),
    ('G', 'H', 5),
]
G.add_weighted_edges_from(aristas)

# Obtener las posiciones de los nodos para el gráfico
posiciones = nx.spring_layout(G)

# Dibujar los nodos
nx.draw_networkx_nodes(G, posiciones, node_size=300, node_color='lightblue')

# Dibujar las aristas
nx.draw_networkx_edges(G, posiciones, width=2, edge_color='gray')

# Dibujar las etiquetas de los nodos
nx.draw_networkx_labels(G, posiciones, font_size=10, font_color='black')

# Dibujar las etiquetas de las aristas (distancias)
etiquetas_aristas = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, posiciones, edge_labels=etiquetas_aristas, font_size=8)

# Mostrar el gráfico
plt.title("Problema del Agente Viajero")
plt.axis('off')
plt.show()




import itertools

n = distance_matrix.shape[0]  # Número de ciudades

# Generar todas las permutaciones posibles de las ciudades
ciudades = np.arange(n)
permutaciones = list(itertools.permutations(ciudades))

# Calcular la distancia para cada permutación
for permutacion in permutaciones:
    permutacion = np.array(permutacion)
    ruta = permutacion + 1  # Ajustar el índice base a 1
    distancia = sum(distance_matrix[permutacion[i], permutacion[i+1]] for i in range(n-1))
    distancia += distance_matrix[permutacion[n-1], permutacion[0]]  # Agregar la distancia de regreso al origen
    print("Ruta:", ruta, "Distancia:", distancia)
  
