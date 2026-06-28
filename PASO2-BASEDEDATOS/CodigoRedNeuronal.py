import os
import time
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (balanced_accuracy_score, accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay,
                             f1_score, precision_score, recall_score, roc_auc_score)

# CARGA Y LIMPIEZA INICIAL
folder_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(folder_path, "dataset_final.csv")
data_df = pd.read_csv(file_path)

# VARIABLE OBJETIVO

data_df['target'] = data_df['on_sentry_list'].astype(int)


# SELECCIÓN DE FEATURES

FEATURE_COLUMNS = [
    'distance_au',
    'velocity_km_s',
    'absolute_magnitude',
    'diametro_km',
    'days_until_approach',
    'year',
    'month',
    'ast_pre_x',
    'ast_pre_y',
    'ast_pre_z',
    'ast_pre_vx',
    'ast_pre_vy',
    'ast_pre_vz',
    'tierra_ca_x',
    'tierra_ca_y',
    'tierra_ca_z',
]

processed_df = data_df[FEATURE_COLUMNS + ['target']].copy()

# Ya no hay columnas categóricas en el conjunto de features
categorical_columns = pd.Index([])
numerical_columns = pd.Index(FEATURE_COLUMNS)

# PREPARACIÓN DE DATOS
X = processed_df.drop(columns=['target'])
y = processed_df['target']
groups = data_df.loc[processed_df.index, 'asteroid_designation']

# ==========================================
# SPLIT TRAIN/TEST AGRUPADO POR ASTEROIDE
# ==========================================
# Mismo motivo que en CodigoCompararModelos.py: on_sentry_list es una
# propiedad del asteroide, no de la fecha de aproximación concreta, y un
# mismo asteroide aparece en varias filas. GroupShuffleSplit evita que el
# mismo asteroide caiga repartido entre train y test (data leakage).
gss = GroupShuffleSplit(n_splits=1, test_size=1/3, random_state=777)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

asteroides_train = set(groups.iloc[train_idx])
asteroides_test = set(groups.iloc[test_idx])
assert len(asteroides_train & asteroides_test) == 0, "Hay leakage entre train y test"
print(f"Asteroides únicos train: {len(asteroides_train)} | test: {len(asteroides_test)}")
print(f"Proporción clase positiva en train: {y_train.mean():.4f} | test: {y_test.mean():.4f}")

# PIPELINE PARA RED NEURONAL
# Las redes neuronales son MUY sensibles al escalado, StandardScaler es preferible.
# Con los vectores ast_pre_*/tierra_ca_* incluidos hay ~4.700 nulos extra
# (asteroides sin posición calculada a T-30 días), por eso la imputación
# por mediana sigue siendo necesaria.
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns)
    ]
)

# Definimos el modelo base de Red Neuronal (MLP)
# class_weight no existe en MLPClassifier; el desbalanceo se compensa
# pasando sample_weight en el fit (ver más abajo) en lugar de un parámetro
# del constructor.
mlp = MLPClassifier(max_iter=1000, random_state=777, early_stopping=True)

pipeline_mlp = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', mlp)
])

# OPTIMIZACIÓN DE HIPERPARÁMETROS (GridSearch)
param_grid = {
    'classifier__hidden_layer_sizes': [(50,), (100,), (50, 25)],
    'classifier__activation': ['relu', 'tanh'],
    'classifier__alpha': [0.0001, 0.01], # Regularización L2
    'classifier__learning_rate_init': [0.001, 0.01]
}

# Usamos scoring='f1' en vez de 'balanced_accuracy': con ~7% de positivos
# queremos optimizar directamente el equilibrio precision/recall de la
# clase minoritaria (en Sentry), que es la que de verdad importa.
print("Iniciando entrenamiento de Red Neuronal...")
grid_search = GridSearchCV(pipeline_mlp, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=3)
start_time = time.time()

# sample_weight para compensar el desbalanceo de clases (MLPClassifier no
# acepta class_weight directamente como RandomForest/LogisticRegression)
sample_weight = np.where(y_train == 1, (y_train == 0).sum() / (y_train == 1).sum(), 1.0)
grid_search.fit(X_train, y_train, classifier__sample_weight=sample_weight)

training_time = time.time() - start_time

# EVALUACIÓN
best_mlp = grid_search.best_estimator_
y_pred = best_mlp.predict(X_test)

bal_acc = balanced_accuracy_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"\n--- RESULTADOS RED NEURONAL ---")
print(f"Mejores parámetros: {grid_search.best_params_}")
print(f"Tiempo de entrenamiento: {training_time:.2f} segundos")
print(f"Balanced Accuracy: {bal_acc:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

if hasattr(best_mlp, "predict_proba"):
    y_proba = best_mlp.predict_proba(X_test)[:, 1]
    print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")

print("\nReporte de clasificación completo:")
print(classification_report(y_test, y_pred, target_names=['No Sentry', 'En Sentry'], zero_division=0))

# GUARDAR MODELO
with open('modelo_nn_sentry.pkl', 'wb') as f:
    pickle.dump(best_mlp, f)

print("\nModelo de Red Neuronal guardado como 'modelo_nn_sentry.pkl'")