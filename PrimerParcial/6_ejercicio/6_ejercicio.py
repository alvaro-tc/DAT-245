import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz

# Paso 2: Lee los datos desde el archivo Excel
data = pd.read_csv('dataset_normalizado.csv')

# Paso 3: Divide los datos en características (X) y etiquetas (y)
X = data.drop('tension_strenght', axis=1)
y = data['tension_strenght']

# Paso 4: Crea una instancia del clasificador de árbol de decisión
clf = DecisionTreeClassifier()

# Paso 5: Ajusta el clasificador utilizando los datos de entrenamiento
clf.fit(X, y)

# Paso 6: Exporta el árbol de decisión a formato DOT
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=X.columns, impurity=False)

# Paso 7: Crea el objeto Graph desde el archivo DOT
graph = graphviz.Source(dot_data)

# Paso 8: Abre el árbol de decisión en una aplicación externa que permita zoom
graph.view()