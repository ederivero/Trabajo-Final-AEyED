import numpy as np
import heapq
from sklearn import datasets

# Definición de la clase Node
class Node:
    def __init__(self, point, left=None, right=None):
        self.point = point
        self.left = left
        self.right = right

class KDTree:
    def __init__(self, points):
        if len(points) == 0:
            raise ValueError("La lista de puntos no puede estar vacía")

        self.k = len(points[0])
        self.root = self.build(points)

    def build(self, points, depth=0):
        if not points:
            return None

        axis = depth % self.k
        points.sort(key=lambda x: x[axis])
        median = len(points) // 2

        return Node(
            point=points[median],
            left=self.build(points[:median], depth + 1),
            right=self.build(points[median + 1:], depth + 1)
        )

    def closest_points(self, target, k=1):
        if self.root is None:
            return []

        closest = [(np.linalg.norm(np.array(self.root.point) - np.array(target)), self.root.point)]
        heapq.heapify(closest)

        def nearest_neighbors(node, depth):
            if node is None:
                return

            axis = depth % self.k
            dist = np.linalg.norm(np.array(node.point) - np.array(target))

            if dist < closest[0][0]:
                heapq.heappush(closest, (dist, node.point))
                if len(closest) > k:
                    heapq.heappop(closest)

            if target[axis] < node.point[axis]:
                nearest_neighbors(node.left, depth + 1)
            else:
                nearest_neighbors(node.right, depth + 1)

        nearest_neighbors(self.root, 0)

        k_closest_points = [point for (_, point) in sorted(closest)]
        return k_closest_points

# Cargar el conjunto de datos de vinos
wine_dataset = datasets.load_wine()
wine_data = wine_dataset.data

# Convertir los datos en una lista de puntos
wine_points = [list(point) for point in wine_data]

# Construir el árbol KD con los datos de vinos
kdtree = KDTree(wine_points)

# Punto de destino para encontrar vecinos cercanos (por ejemplo, el primer punto del conjunto de datos)
target_point = wine_points[0]

# Encuentra los 5 vecinos más cercanos al punto de destino
k_neighbors = kdtree.closest_points(target_point, k=5)
print("Los 5 vecinos más cercanos al punto de destino son:")
for neighbor in k_neighbors:
    print(neighbor)
