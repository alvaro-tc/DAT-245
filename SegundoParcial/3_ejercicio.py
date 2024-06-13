import numpy as np
import random
import matplotlib.pyplot as plt

def distancia(ciudad1, ciudad2):
    return np.sqrt((ciudad1[0] - ciudad2[0])**2 + (ciudad1[1] - ciudad2[1])**2)

def calcular_distancia_total(ruta, ciudades):
    distancia_total = 0
    for i in range(len(ruta) - 1):
        distancia_total += distancia(ciudades[ruta[i]], ciudades[ruta[i + 1]])
    distancia_total += distancia(ciudades[ruta[-1]], ciudades[ruta[0]])  # Volver al punto de inicio
    return distancia_total

def generar_vecinos(ruta):
    vecinos = []
    for i in range(len(ruta)):
        for j in range(i + 1, len(ruta)):
            vecino = ruta[:]
            vecino[i], vecino[j] = vecino[j], vecino[i]
            vecinos.append(vecino)
    return vecinos

def buscar_mejor_vecino(ruta, ciudades):
    mejor_vecino = ruta
    mejor_distancia = calcular_distancia_total(ruta, ciudades)
    vecinos = generar_vecinos(ruta)
    for vecino in vecinos:
        distancia_actual = calcular_distancia_total(vecino, ciudades)
        if distancia_actual < mejor_distancia:
            mejor_vecino = vecino
            mejor_distancia = distancia_actual
    return mejor_vecino, mejor_distancia

def busqueda_local(ciudades, iteraciones):
    ruta_actual = list(range(len(ciudades)))
    random.shuffle(ruta_actual)
    mejor_ruta = ruta_actual
    mejor_distancia = calcular_distancia_total(ruta_actual, ciudades)
    historial_mejor_distancia = [mejor_distancia]

    for i in range(iteraciones):
        vecino, distancia_vecino = buscar_mejor_vecino(ruta_actual, ciudades)
        if distancia_vecino < mejor_distancia:
            mejor_ruta = vecino
            mejor_distancia = distancia_vecino
            historial_mejor_distancia.append(mejor_distancia)
        ruta_actual = vecino

    return mejor_ruta, mejor_distancia, historial_mejor_distancia

num_ciudades = 20
ciudades = np.random.rand(num_ciudades, 2)

# Ejecutar búsqueda local
mejor_ruta, mejor_distancia, historial_mejor_distancia = busqueda_local(ciudades, 1000)

# Mostrar resultados
print("Mejor ruta:", mejor_ruta)
print("Mejor distancia:", mejor_distancia)

plt.plot(historial_mejor_distancia)
plt.xlabel('Iteraciones')
plt.ylabel('Distancia')
plt.title('Reducción de la distancia durante la búsqueda local')
plt.show()

# Visualizar la mejor ruta
ruta_x = [ciudades[ciudad][0] for ciudad in mejor_ruta] + [ciudades[mejor_ruta[0]][0]]
ruta_y = [ciudades[ciudad][1] for ciudad in mejor_ruta] + [ciudades[mejor_ruta[0]][1]]

plt.figure()
plt.plot(ruta_x, ruta_y, 'o-', label='Mejor ruta')
plt.scatter(ciudades[:, 0], ciudades[:, 1], color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Mejor ruta encontrada')
plt.legend()
plt.show()
