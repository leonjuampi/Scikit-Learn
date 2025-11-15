import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

#nos traemos el dataset de propinas
data = sns.load_dataset('tips')

print("--- Informacion del Dataset 'Tips' ---")
data.info()
#todos los campos estan completos, no hay datos faltantes

print("\n--- Primeras 5 Filas ---")
print(data.head())

print("\n--- cuantos fumadores y no fumadores hay? ---")
print(data['smoker'].value_counts())


y = data['smoker']
# sacamos smoker porque es lo que queremos predecir
X = data.drop('smoker', axis=1)


# 'total_bill', 'tip' y 'size' ya son numeros.
# Convertimos 'sex', 'day' y 'time' 
X = pd.get_dummies(X, columns=['sex', 'day', 'time'], drop_first=True)

print("\n--- convertidas a numeros ---")
print(X.head())



print("\n--- Buscamos los Mejores Hiperparámetros ---")

# parametros que vamos a usar, max_depth es la cantidad de divisiones y min_samples_split es la cantidad minima de muestras para hacer una division
param_grid = {
    'min_samples_split': [2, 5, 10, 20],
    'max_depth': [3, 4, 5, None] # 'None' significa que el arbol crece sin limite
}

# n_jobs=-1 usa todos los procesadores para que vuele
grid = GridSearchCV(DecisionTreeClassifier(random_state=42), 
                      param_grid, 
                      cv=5, 
                      n_jobs=-1)

# Entrenamos TODOS los datos
grid.fit(X, y)


print(f"Mejores Parametros encontrados: {grid.best_params_}")
print(f"Mejor Score (Accuracy) de Cross-Validation: {grid.best_score_:.4f}")


print("\n--- Entrenando y Evaluando Modelo Final ---")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12122135)

#guardamos un diccionario con los mejores parametros
best_params = grid.best_params_
# aca creamos el modelo final con los mejores parametros
final_model = DecisionTreeClassifier(
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    random_state=42
)

#Entrenamos con Train
final_model.fit(X_train, y_train)

#Hacemos predicciones de Test
y_pred = final_model.predict(X_test)

#Sacamos el Score final (Accuracy)
print(f"Score (Accuracy) del Modelo Final en Test: {accuracy_score(y_test, y_pred):.4f}")

print("\nReporte de Clasificacion:")
print(classification_report(y_test, y_pred))

print("\nMatriz de Confusion:")
print(confusion_matrix(y_test, y_pred))


plt.figure(figsize=(10, 10)) #este es el tamaño
plot_tree(
    final_model,
    feature_names=X.columns.tolist(), # Nombres de las features
    class_names=final_model.classes_.tolist(), # Nombres
    filled=True,  # da colores
    rounded=True  # Bordes redondos
)
plt.title(f"Árbol de Decision para Predecir Fumadores (max_depth={best_params['max_depth']})")
plt.show()
