# Regressão Linear Simples

# dados
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Horas de estudo
y = [41.6, 44.9, 54.5, 60.9, 61.1, 67.7, 70.6, 75.0, 82.4, 85.2]  # Notas

# calcular médias
n = len(x)
x_media = sum(x) / n
y_media = sum(y) / n

# calcular coeficiente angular (b)
numerador = sum((x[i] - x_media) * (y[i] - y_media) for i in range(n))
denominador = sum((x[i] - x_media) ** 2 for i in range(n))
b = numerador / denominador

# calcular intercepto (a)
a = y_media - b * x_media

# resultados
print(f"Coeficiente angular (b): {b:.4f}")
print(f"Intercepto (a): {a:.4f}")