# imports

import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, TargetEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle
import pandas as pd
import ast
import re
from sklearn.model_selection import RandomizedSearchCV

# Cargamos el archivo

folder_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(folder_path, "dataset_final.csv")

data_df = pd.read_csv(file_path) 



#LO COMENTO PORQUE ENSUCIA LA SOLUCION Y ES INFORMATIVO

# Exploración de los datos
print('Tamaño de la tabla de datos:')
print('===============================')
print(data_df.shape)
print()

print('Tipos de atributos:')
print('================================')
data_df.info()

print()

print('Valores faltantes por atributo:')
print('======================================')
print(data_df.isnull().sum())

print()

print('Proporción de valores faltantes por atributo:')
print('======================================')
print(data_df.isnull().mean())
print()


"""
#ESTO LO COMENTO PORQUE PUEDE QUE LO VUELVA A UTILIZAR

# 1. Comprobar cuántas veces aparece cada clase
print('Recuento de clases en la columna "panic_level":')
print('===========================================')
print(data_df['panic_level'].value_counts())
print()

# 2. Comprobar la proporción de cada clase
print('Proporción de cada clase:')
print('=========================')
print(data_df['panic_level'].value_counts(normalize=True))

# Por supuesto está desvalancado
"""
# parece que las variables sentry_impact_prob, sentry_torino_scale, sentry_palermo_scale y sentry_diameter_km estan muy vacías

# ELIMINO LAS FILAS DE PANIC LEVEL CON CANTIDADES MENORES A 10 es decir la clase 9
counts = data_df['panic_level'].value_counts()
clases_validas = counts[counts >= 10].index
data_df = data_df[data_df['panic_level'].isin(clases_validas)]

# Identificar columnas constantes, excepto 'vector_posicion_tierra'
constant_columns = [col for col in data_df.columns if data_df[col].nunique() == 1]

# Eliminar las columnas constantes del dataframe
processed_df = data_df.drop(columns=constant_columns)













# Voy a eliminar las columnas categóricas ya que añaden mucha complejidad al modelo, y no aportan información relevante para la clasificación 
processed_df = data_df.drop(columns=['asteroid_designation', 'asteroid_fullname', 'close_approach_date'], errors='ignore')
















# Eliminamos la columna inutil de panic_verdict
processed_df = processed_df.drop(columns=['panic_verdict'], errors='ignore')
'''
# LO COMENTO PORQUE SOLO ENSUCIA LA SOLUCION Y ES INFORMATIVO

# Mostrar las columnas eliminadas
print(f"Columnas eliminadas (constantes): {constant_columns}")
'''

# Paso 1: Identificar variables categóricas
categorical_columns = processed_df.select_dtypes(include=['object']).columns
# Y numéricas, quitando la variable objetivo para que no intente escalarla
numerical_columns = processed_df.select_dtypes(include=['int64', 'float64']).columns.drop('panic_level', errors='ignore')
# Paso 2: Contar las categorías únicas por variable
cardinality = processed_df[categorical_columns].nunique()

#ES INFORMATIVO, LO COMENTO PARA NO ENSUCIAR LA SOLUCION
print("\n\nCardinalidad de las variables categóricas:")
print(cardinality)


# OBSERVO LOS VALORES ÚNICOS DEL PANIC LEVEL

# Verificar valores únicos en panic_level (es del 0 al 9))
print("\nValores únicos en panic_level:")
print(processed_df['panic_level'].unique())
print("Recuento por clase:")
print(processed_df['panic_level'].value_counts().sort_index())



# Opciones de escalado e imputación
scalers = {
    "standard": StandardScaler(),
    "minmax": MinMaxScaler(),
    "robust": RobustScaler()
}

imputers = {
    "mean": SimpleImputer(strategy="mean"),
    "median": SimpleImputer(strategy="median")
}

# Como el dataset se encuentra desbalanceado debemos incluir `class_weight='balanced' en el clasifier

# Definir pipeline base
def create_pipeline(scaler, imputer):
    numerical_transformer = Pipeline(steps=[
        ('imputer', imputer),
        ('scaler', scaler)
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('target_enc', TargetEncoder(target_type='continuous')) 
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(class_weight='balanced'))
    ])

    return pipeline

# Creamos la instancia del pipeline
pipeline_base = create_pipeline(scaler=StandardScaler(), imputer=SimpleImputer())

# Definimos qué queremos probar
param_grid = [
    {
        'classifier': [LogisticRegression(class_weight='balanced', max_iter=50000)],
        'preprocessor__num__scaler': [StandardScaler(), MinMaxScaler()],
        'classifier__C': [0.1, 1, 10],
        'classifier__penalty': ['l2']
    },
    {
        'classifier': [RandomForestClassifier(class_weight='balanced')],
        'classifier__min_samples_leaf': [10, 20, 50],
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10],
        'classifier__max_features': ['sqrt', 'log2']
    }
]


grid = GridSearchCV(
    pipeline_base, 
    param_grid, 
    cv=5, 
    scoring='f1_weighted', 
    n_jobs=-1, 
    verbose=1
)


# Codificamos la salida separando los datos en train/test (Holdout)

X = processed_df.drop(columns=['panic_level'])
y = processed_df['panic_level']  # Definir la variable objetivo

# Seleccionados train/test en modo stratify, ya que el dataset se encuentra desbalanceado
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=1/3, random_state=777)
# hago siempre el mismo random_state para que siemrpe me de los mismos resultados

grid = GridSearchCV(pipeline_base, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
# Entrenamos (aquí es donde se prueban todos los modelos)
grid.fit(X_train, y_train)

# Resultado final
print("******* Los mejores hiperparámetros KNN encontrados son:")
print(grid.best_params_)

mejor_modeloknn = grid.best_estimator_
y_pred = mejor_modeloknn.predict(X_test)

cv_scoresknn = cross_val_score(mejor_modeloknn, X_train, y_train, cv=5, scoring='balanced_accuracy', n_jobs=-1)

# Mostrar los resultados
print("Media de validación cruzada con KNN, GridSearchCV:", cv_scoresknn.mean())

# HPO: Búsqueda de hiperparámetros óptimos
param_grid2 = {
        'classifier': [DecisionTreeClassifier(random_state=42, class_weight='balanced')],
        'preprocessor__num__imputer': [SimpleImputer(strategy='mean'), SimpleImputer(strategy='median')],
        'preprocessor__num__scaler': [StandardScaler(), MinMaxScaler(), RobustScaler()],
        'classifier__max_depth': [3, 5, 10, 15],
        'classifier__min_samples_split': [10, 20, 50],
        'classifier__min_samples_leaf': [5, 10, 20, 50],
        'classifier__criterion': ['gini', 'entropy']
    }

grid_search2 = GridSearchCV(
    create_pipeline(MinMaxScaler(), SimpleImputer(strategy="mean")), 
    param_grid2,
    scoring='balanced_accuracy',
    n_jobs=-1,
    verbose=1
)


start_time = time.time()
grid_search2.fit(X_train, y_train)
tiempo = time.time() - start_time

best_DTC = grid_search2.best_estimator_

print("******* Los mejores hiperparámetros DTC encontrados son:")
print(grid_search2.best_params_)

y_pred_best = best_DTC.predict(X_test)
DTC_Bal_acc = balanced_accuracy_score(y_test, y_pred_best)
DTC_acc = accuracy_score(y_test, y_pred_best)

# Realizar validación cruzada con 5 fbalancedolds en los datos de entrenamiento
cv_scoresDTC = cross_val_score(best_DTC, X_train, y_train, cv=5, scoring="balanced_accuracy", n_jobs=-1)

# Mostrar los resultados
print("Media de validación cruzada, hiperparámetros DTC:", cv_scoresDTC.mean())

#compruebo cual es el mejor
if cv_scoresknn < cv_scoresDTC:
    print("El mejor modelo es el DTC")
    ElMejorModelo = best_DTC
else:
    print("El mejor modelo es el KNN")
    ElMejorModelo = mejor_modeloknn

# Guardar el modelo entrenado
with open('modelo_panico_asteroides.pkl', 'wb') as f:
    pickle.dump(ElMejorModelo, f)

print("Modelo guardado como 'modelo_panico_asteroides.pkl'")
'''
# Predicción y métricas
rf_pred_best = dtc_pipeline.predict(X_test)
RF_Bal_acc = balanced_accuracy_score(y_test, rf_pred_best)
RF_acc = accuracy_score(y_test, rf_pred_best)

# Calcular accuracy en train para detectar overfitting
y_pred_train = dtc_pipeline.predict(X_train)
RF_acc_train = accuracy_score(y_train, y_pred_train)
RF_bal_acc_train = balanced_accuracy_score(y_train, y_pred_train)

print(f"Accuracy en train: {RF_acc_train:.4f}")
print(f"Balanced Accuracy en train: {RF_bal_acc_train:.4f}")
print(f"Accuracy en test: {RF_acc:.4f}")
print(f"Balanced Accuracy en test: {RF_Bal_acc:.4f}")

# Validación cruzada ligera (reducida a 2 folds por clases desbalanceadas)
cv_scores = cross_val_score(dtc_pipeline, X_train, y_train, cv=2, scoring='balanced_accuracy', n_jobs=3)

# Evaluación
matrizConfusion = confusion_matrix(y_test, rf_pred_best)

print("BalancedAcc:", RF_Bal_acc)
print(f"\nBalanced Accuracy con RF multiclase: {RF_Bal_acc}")
print("--------------------------------------")
print("Acc:", RF_acc)
print(f"\nAcc con RF multiclase: {RF_acc}")
print("--------------------------------------")
print("MatrizConf (10x10 para clases 0-9):", matrizConfusion)
print("--------------------------------------")
print("Tiempo de entrenamiento con RF multiclase:", dtc_training_time)
print("--------------------------------------\n\n")

# Para multiclase, calculamos precision, recall y F1 macro-average
from sklearn.metrics import classification_report
print("Reporte de clasificación (macro-average):")
print(classification_report(y_test, rf_pred_best, labels=range(10), target_names=[f'Clase {i}' for i in range(10)], zero_division=0))

'''