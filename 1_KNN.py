import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from collections import Counter

# Implementación de la distancia euclidiana
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Implementación del algoritmo KNN
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Calcular las distancias entre x y todos los puntos de entrenamiento
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Obtener los índices de los k puntos más cercanos
        k_indices = np.argsort(distances)[:self.k]
        # Obtener las etiquetas de los k puntos más cercanos
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Realizar una votación para determinar la etiqueta más común
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Cargar el conjunto de datos Wine
wine = load_wine()
X = wine.data
y = wine.target

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear un clasificador KNN con K=5
knn = KNN(k=5)

# Entrenar el modelo
knn.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = knn.predict(X_test)
print(y_pred)
# Calcular la precisión de las predicciones
accuracy = np.mean(y_pred == y_test)
print(f'Precisión del clasificador KNN: {accuracy * 100:.2f}%')