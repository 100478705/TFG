import os
import time
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (balanced_accuracy_score, accuracy_score, confusion_matrix, 
                             classification_report, ConfusionMatrixDisplay)

# CARGA Y LIMPIEZA INICIAL
folder_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(folder_path, "dataset_final.csv")
data_df = pd.read_csv(file_path)

# Eliminar clases con menos de 10 muestras (clase 9)
counts = data_df['panic_level'].value_counts()
clases_validas = counts[counts >= 10].index
data_df = data_df[data_df['panic_level'].isin(clases_validas)]

# Eliminar columnas constantes y categóricas irrelevantes
constant_columns = [col for col in data_df.columns if data_df[col].nunique() == 1]
processed_df = data_df.drop(columns=constant_columns)
processed_df = processed_df.drop(columns=['asteroid_designation', 'asteroid_fullname', 'close_approach_date', 'panic_verdict'], errors='ignore')

# DEFINICIÓN DE COLUMNAS
categorical_columns = processed_df.select_dtypes(include=['object']).columns
numerical_columns = processed_df.select_dtypes(include=['int64', 'float64']).columns.drop('panic_level', errors='ignore')

# PREPARACIÓN DE DATOS
X = processed_df.drop(columns=['panic_level'])
y = processed_df['panic_level']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=1/3, random_state=777)

# PIPELINE PARA RED NEURONAL
# Las redes neuronales son MUY sensibles al escalado, StandardScaler es preferible.
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
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

# Definimos el modelo base de Red Neuronal (MLP)
mlp = MLPClassifier(max_iter=1000, random_state=777, early_stopping=True)

pipeline_mlp = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', mlp)
])

# OPTIMIZACIÓN DE HIPERPARÁMETROS (GridSearch)
# Definimos parámetros típicos para una Red Neuronal
param_grid = {
    'classifier__hidden_layer_sizes': [(50,), (100,), (50, 25)],
    'classifier__activation': ['relu', 'tanh'],
    'classifier__alpha': [0.0001, 0.01], # Regularización L2
    'classifier__learning_rate_init': [0.001, 0.01]
}

print("Iniciando entrenamiento de Red Neuronal...")
grid_search = GridSearchCV(pipeline_mlp, param_grid, cv=3, scoring='balanced_accuracy', n_jobs=-1, verbose=3)
start_time = time.time()
grid_search.fit(X_train, y_train)
training_time = time.time() - start_time

# EVALUACIÓN
best_mlp = grid_search.best_estimator_
y_pred = best_mlp.predict(X_test)

bal_acc = balanced_accuracy_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print(f"\n--- RESULTADOS RED NEURONAL ---")
print(f"Mejores parámetros: {grid_search.best_params_}")
print(f"Tiempo de entrenamiento: {training_time:.2f} segundos")
print(f"Balanced Accuracy: {bal_acc:.4f}")
print(f"Accuracy: {acc:.4f}")

# GUARDAR MODELO
with open('modelo_nn_panico.pkl', 'wb') as f:
    pickle.dump(best_mlp, f)

print("\nModelo de Red Neuronal guardado como 'modelo_nn_panico.pkl'")