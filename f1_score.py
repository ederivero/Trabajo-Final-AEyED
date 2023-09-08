import numpy as np

def calculate_f1_score(nearest_points, resulting_points):
    def vector_to_tuple(vector):
        return tuple(vector)

    # Convertir las listas de puntos a conjuntos de tuplas para facilitar las operaciones
    nearest_set = set(map(vector_to_tuple, nearest_points))
    resulting_set = set(map(vector_to_tuple, resulting_points))

    # Calcular TP, FP y FN
    true_positives = len(nearest_set.intersection(resulting_set))
    false_positives = len(resulting_set - nearest_set)
    false_negatives = len(nearest_set - resulting_set)

    # Calcular el puntaje F1
    if true_positives == 0:
        return 0.0  # Evitar la división por cero
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score

# Ejemplo de uso
nearest_points = [
    [12.37, 1.63, 2.3, 24.5, 88.0, 2.22, 2.45, 0.4, 1.9, 2.12, 0.89, 2.78, 342.0],
    [12.43, 1.53, 2.29, 21.5, 86.0, 2.74, 3.15, 0.39, 1.77, 3.94, 0.69, 2.84, 352.0],
    [11.62, 1.99, 2.28, 18.0, 98.0, 3.02, 2.26, 0.17, 1.35, 3.25, 1.16, 2.96, 345.0],
    [12.52, 2.43, 2.17, 21.0, 88.0, 2.55, 2.27, 0.26, 1.22, 2.0, 0.9, 2.78, 325.0],
    [12.42, 1.61, 2.19, 22.5, 108.0, 2.0, 2.09, 0.34, 1.61, 2.06, 1.06, 2.96, 345.0]
]

resulting_points = [
    [12.37, 1.63, 2.3, 24.5, 88.0, 2.22, 2.45, 0.4, 1.9, 2.12, 0.89, 2.78, 342.0],
    [11.41, 0.74, 2.5, 21.0, 88.0, 2.48, 2.01, 0.42, 1.44, 3.08, 1.1, 2.31, 434.0],
    [12.43, 1.53, 2.29, 21.5, 86.0, 2.74, 3.15, 0.39, 1.77, 3.94, 0.69, 2.84, 352.0],
    [12.42, 1.61, 2.19, 22.5, 108.0, 2.0, 2.09, 0.34, 1.61, 2.06, 1.06, 2.96, 345.0],
    [13.05, 1.77, 2.1, 17.0, 107.0, 3.0, 3.0, 0.28, 2.03, 5.04, 0.88, 3.35, 885.0],
]

f1 = calculate_f1_score(nearest_points, resulting_points)
print("Puntaje F1:", f1)