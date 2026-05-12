# PARA VER LOS PARÁMETROS DEL MODELO GANADOR
"""import pickle
import os

# 1. Obtener la ruta de la carpeta donde está este script
folder_path = os.path.dirname(os.path.abspath(__file__))
# 1. Cargamos el modelo guardado
with open(os.path.join(folder_path, 'modelo_panico_asteroides.pkl'), 'rb') as f:
    modelo_cargado = pickle.load(f)

# 2. Extraer todos los parámetros
# Como es un Pipeline, verás nombres como 'classifier__n_estimators' o 'preprocessor__num__scaler'
parametros = modelo_cargado.get_params()

print("--- CONFIGURACIÓN COMPLETA DEL MODELO ---")
# Filtramos para ver solo lo importante (el clasificador y el preprocesador)
for llave, valor in parametros.items():
    if 'classifier__' in llave or 'preprocessor__' in llave:
        print(f"{llave}: {valor}")

# 3. Si quieres ver solo el nombre del algoritmo que ganó:
print("\n--- ALGORITMO GANADOR ---")
print(modelo_cargado.named_steps['classifier'])"""







import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (balanced_accuracy_score, accuracy_score, confusion_matrix, 
                             classification_report)

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS Y CARPETAS
# ==========================================
folder_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(folder_path, "dataset_final.csv")
charts_folder = os.path.join(folder_path, "graficas")

# Crear carpeta de gráficas si no existe
if not os.path.exists(charts_folder):
    os.makedirs(charts_folder)
    print(f"Carpeta creada con éxito: {charts_folder}")

# ==========================================
# 2. CARGA Y LIMPIEZA DE DATOS (Exactamente igual que en el entrenamiento)
# ==========================================
print("Cargando y preparando los datos...")
data_df = pd.read_csv(file_path)

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
# Separar X e y
X = processed_df.drop(columns=['panic_level'])
y = processed_df['panic_level']

# IMPORTANTE: Usamos el mismo random_state y test_size para aislar el mismo conjunto de Test
_, X_test, _, y_test = train_test_split(X, y, stratify=y, test_size=1/3, random_state=777)

# ==========================================
# 3. CARGA DE MODELOS PREENTRENADOS
# ==========================================
print("\nCargando modelos desde el directorio raíz...")

with open(os.path.join(folder_path, 'modelo_nn_panico.pkl'), 'rb') as f:
    modelo_nn = pickle.load(f)

with open(os.path.join(folder_path, 'modelo_panico_asteroides.pkl'), 'rb') as f:
    modelo_arbol_knn = pickle.load(f)

# ==========================================
# 4. FUNCIÓN DE EVALUACIÓN Y GRAFICADO
# ==========================================
def evaluar_y_graficar(modelo, nombre_modelo, X_test, y_test, charts_folder):
    print(f"\n--- Evaluando: {nombre_modelo} ---")
    
    # Predecir (el pipeline interno del modelo se encarga de escalar e imputar automáticamente)
    y_pred = modelo.predict(X_test)
    
    # Métricas
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Accuracy regular: {acc:.4f}")

    # --- Gráfica 1: Matriz de Confusión ---
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f'Matriz de Confusión - {nombre_modelo}')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.savefig(os.path.join(charts_folder, f'matriz_confusion_{nombre_modelo.replace(" ", "_")}.png'))
    plt.close()

    # --- Gráfica 2: Reporte de Clasificación ---
    plt.figure(figsize=(10, 6))
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap='RdYlGn')
    plt.title(f'Métricas por Clase - {nombre_modelo}')
    plt.savefig(os.path.join(charts_folder, f'reporte_clasificacion_{nombre_modelo.replace(" ", "_")}.png'))
    plt.close()

# ==========================================
# 5. EJECUCIÓN
# ==========================================
# Evaluamos el modelo anterior (Árbol de Decisión o KNN)
evaluar_y_graficar(modelo_arbol_knn, "Modelo_Clasico", X_test, y_test, charts_folder)

# Evaluamos la Red Neuronal
evaluar_y_graficar(modelo_nn, "Red_Neuronal", X_test, y_test, charts_folder)

# --- Gráfica Exclusiva 3: Curva de Pérdida de la Red Neuronal ---
# Como el modelo está guardado como un Pipeline, tenemos que extraer el clasificador ('classifier')
try:
    clasificador_nn = modelo_nn.named_steps['classifier']
    if hasattr(clasificador_nn, 'loss_curve_'):
        plt.figure(figsize=(8, 5))
        plt.plot(clasificador_nn.loss_curve_)
        plt.title('Curva de Pérdida - Red Neuronal')
        plt.xlabel('Iteraciones')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(charts_folder, 'curva_perdida_Red_Neuronal.png'))
        plt.close()
        print(f"\nCurva de pérdida de la Red Neuronal guardada con éxito.")
except Exception as e:
    print(f"\nNo se pudo generar la curva de pérdida. Motivo: {e}")

print(f"\n¡Proceso finalizado! Todas las gráficas se han guardado en: {charts_folder}")