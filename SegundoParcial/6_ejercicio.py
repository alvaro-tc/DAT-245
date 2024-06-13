import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt

# Definir el grafo
G = nx.Graph()
G.add_edges_from([('A', 'B', {'weight': 7}), ('A', 'C', {'weight': 9}), ('A', 'D', {'weight': 8}), ('A', 'E', {'weight': 20}),
                  ('B', 'D', {'weight': 4}), ('B', 'C', {'weight': 10}), ('B', 'E', {'weight': 11}), 
                  ('D', 'C', {'weight': 15}), ('D', 'E', {'weight': 17}),
                  ('E', 'C', {'weight': 5})])

# Definir la posición de los nodos
pos = {'A': (0.5, 1.1), 'B': (0.25, 0.6), 'C': (0.75, 0.6), 'D': (0.35, 0), 'E': (0.65, 0)}

# Dibujar el grafo con las distancias
edge_labels = nx.get_edge_attributes(G, 'weight')
node_size = 2000
font_size = 16
nx.draw(G, pos, with_labels=True, labels={'A':'A', 'B':'B', 'C':'C', 'D':'D', 'E':'E'}, node_size=node_size, font_size=font_size)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=font_size)

plt.show()

# Función para calcular la distancia total de una ruta
def calcular_distancia_total(ruta, G):
    distancia_total = 0
    for i in range(len(ruta) - 1):
        distancia_total += G[ruta[i]][ruta[i + 1]]['weight']
    distancia_total += G[ruta[-1]][ruta[0]]['weight']  # Volver al punto de inicio
    return distancia_total

# Definir la clase del individuo
class Individuo:
    def __init__(self, genes):
        self.genes = genes
        self.fitness = self.calcular_fitness()

    def calcular_fitness(self):
        return calcular_distancia_total(self.genes, G)

# Definir la función para generar la población inicial
def generar_poblacion_inicial(tamano_poblacion, nodos):
    poblacion = []
    for _ in range(tamano_poblacion):
        genes = nodos[1:]  # Excluir el nodo 'A'
        random.shuffle(genes)
        genes = ['A'] + genes  # Asegurar que 'A' sea el primer nodo
        poblacion.append(Individuo(genes))
    return poblacion

# Definir la función de selección por torneo
def seleccion_por_torneo(poblacion, k=2):
    padres = random.sample(poblacion, k)
    padres.sort(key=lambda x: x.fitness)
    return padres[0]

# Definir la función de cruce de orden (OX)
def cruce_ox(padre1, padre2):
    size = len(padre1.genes)
    start, end = sorted(random.sample(range(1, size), 2))  # Excluir el índice 0
    hijo1_genes = ['A'] + [None] * (size - 1)
    hijo1_genes[start:end] = padre1.genes[start:end]

    hijo2_genes = [gen for gen in padre2.genes if gen not in hijo1_genes]
    hijo1_genes = ['A'] + [hijo2_genes.pop(0) if gen is None else gen for gen in hijo1_genes[1:]]

    hijo1 = Individuo(hijo1_genes)
    return hijo1

# Definir la función de mutación
def mutacion(individuo, tasa_mutacion):
    genes_mutados = individuo.genes[:]
    for i in range(1, len(genes_mutados)):  # Comenzar desde el índice 1
        if random.random() < tasa_mutacion:
            j = random.randint(1, len(genes_mutados) - 1)  # Excluir el índice 0
            genes_mutados[i], genes_mutados[j] = genes_mutados[j], genes_mutados[i]
    return Individuo(genes_mutados)
def algoritmo_genetico(nodos, num_generaciones, tamano_poblacion, tasa_mutacion):
    # Generar la población inicial
    poblacion = generar_poblacion_inicial(tamano_poblacion, nodos)

    # Evolucionar la población
    for _ in range(num_generaciones):
        nueva_poblacion = []
        while len(nueva_poblacion) < tamano_poblacion:
            padre1 = seleccion_por_torneo(poblacion)
            padre2 = seleccion_por_torneo(poblacion)
            hijo = cruce_ox(padre1, padre2)
            hijo = mutacion(hijo, tasa_mutacion)
            nueva_poblacion.append(hijo)
        poblacion = nueva_poblacion

    # Devolver el mejor individuo
    mejor_individuo = min(poblacion, key=lambda x: x.fitness)
    return mejor_individuo

# Ejemplo de uso
nodos = list(G.nodes)
num_generaciones = 1000
tamano_poblacion = 50
tasa_mutacion = 0.1

# Asegurar que 'A' sea el primer nodo
nodos.remove('A')
nodos.insert(0, 'A')

mejor_individuo = algoritmo_genetico(nodos, num_generaciones, tamano_poblacion, tasa_mutacion)
print("Mejor individuo:", mejor_individuo.genes)
print("Mejor distancia:", mejor_individuo.fitness)

# Visualizar la mejor ruta
ruta_x = [pos[nodo][0] for nodo in mejor_individuo.genes] + [pos[mejor_individuo.genes[0]][0]]
ruta_y = [pos[nodo][1] for nodo in mejor_individuo.genes] + [pos[mejor_individuo.genes[0]][1]]

plt.figure()
plt.plot(ruta_x, ruta_y, 'o-', label='Mejor ruta')
plt.scatter([pos[nodo][0] for nodo in G.nodes], [pos[nodo][1] for nodo in G.nodes], color='red')
nx.draw_networkx_labels(G, pos)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Mejor ruta encontrada')
plt.legend()
plt.show()