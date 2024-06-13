import csv
import numpy as np
import matplotlib.pyplot as plt

def leer_csv(file_path):
    with open(file_path, newline='') as csvfile:
        data = list(csv.reader(csvfile, delimiter=','))
        encabezados = data[0]
        data = data[1:]
    return encabezados, data

# Leemos los datos
encabezados, data = leer_csv('Iris.csv')
data = np.array(data)

# Extraer características y etiquetas
X = data[:, 1:5].astype(float)  # Características
y = data[:, 5]  # Etiquetas

# Normalizar las características
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Convertir etiquetas a valores numéricos
label_to_int = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
y = np.array([label_to_int[label] for label in y])

# Convertir etiquetas a one-hot encoding
y_one_hot = np.zeros((y.size, y.max() + 1))
y_one_hot[np.arange(y.size), y] = 1

# Función de activación escalón (Heaviside)
def funcion_escalon(x):
    return np.where(x >= 0, 1, 0)

# Inicialización de pesos
def iniciar_pesos(tamano_entrada, tamano_capa_oculta, tamano_salida):
    np.random.seed(42) 
    W1 = np.random.randn(tamano_entrada, tamano_capa_oculta)
    b1 = np.zeros((1, tamano_capa_oculta))
    W2 = np.random.randn(tamano_capa_oculta, tamano_salida)
    b2 = np.zeros((1, tamano_salida))
    return W1, b1, W2, b2

# Propagación hacia adelante
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = funcion_escalon(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = funcion_escalon(Z2)
    return Z1, A1, Z2, A2

# Propagación hacia atrás
def backward_propagation(X, y, Z1, A1, Z2, A2, W1, W2, b1, b2, tasa_aprendizaje):
    m = X.shape[0]
    dZ2 = A2 - y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * funcion_escalon(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    W1 -= tasa_aprendizaje * dW1
    b1 -= tasa_aprendizaje * db1
    W2 -= tasa_aprendizaje * dW2
    b2 -= tasa_aprendizaje * db2

    return W1, b1, W2, b2

# Entrenamiento de la red neuronal
def entrenar_red_neuronal(X, y, tamano_capa_oculta, tasa_aprendizaje, epocas):
    tamano_entrada = X.shape[1]
    tamano_salida = y.shape[1]

    W1, b1, W2, b2 = iniciar_pesos(tamano_entrada, tamano_capa_oculta, tamano_salida)
    perdidas = []

    for epoch in range(epocas):
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        W1, b1, W2, b2 = backward_propagation(X, y, Z1, A1, Z2, A2, W1, W2, b1, b2, tasa_aprendizaje)
        
        if epoch % 100 == 0:
            perdida = np.mean(np.square(y - A2))
            perdidas.append(perdida)
            print(f'Epoch {epoch}, perdida: {perdida}')

    return W1, b1, W2, b2, perdidas

def predecir(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    return np.argmax(A2, axis=1)

# Entrenar la red neuronal
tamano_capa_oculta = 5
tasa_aprendizaje = 0.2
epocas = 500
W1, b1, W2, b2, perdidas = entrenar_red_neuronal(X, y_one_hot, tamano_capa_oculta, tasa_aprendizaje, epocas)

# Evaluar la precisión del modelo
predictions = predecir(X, W1, b1, W2, b2)
precision = np.mean(np.argmax(y_one_hot, axis=1) == predictions)
print(f'precision: {precision}')

# Graficar la reducción del error
plt.plot(range(0, epocas, 100), perdidas)
plt.xlabel('Epoca')
plt.ylabel('perdida')
plt.title('Reducción del error durante el entrenamiento')
plt.show()
