# imports

import os
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, TargetEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (balanced_accuracy_score, accuracy_score, confusion_matrix,
                              f1_score, precision_score, recall_score, roc_auc_score,
                              classification_report)
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


# ==========================================
# CAMBIO DE VARIABLE OBJETIVO
# ==========================================
# panic_level fue descartada: en el analisis exploratorio se vio que
# correlaciona ~-0.5 con distance_au de forma casi perfectamente
# monotona, y prácticamente 0 con el diametro real o con
# sentry_impact_prob. Es decir, no refleja el riesgo fisico real
# (tamaño + probabilidad de impacto), sino sobre todo la distancia,
# y parece construida con una formula, no con un criterio de riesgo
# solido.
#
# Variable objetivo nueva: on_sentry_list. Es un dato real de la NASA/JPL,
# no inventado por nosotros: indica si el asteroide esta en la lista de
# seguimiento de riesgo de impacto Sentry. Es un problema de clasificacion
# binaria, desbalanceado (~7% positivos) y no trivial (ninguna variable
# individual correlaciona con ella por encima de 0.33).
data_df['target'] = data_df['on_sentry_list'].astype(int)

print('Recuento de clases en la variable objetivo "on_sentry_list":')
print('===========================================')
print(data_df['target'].value_counts())
print()
print('Proporción de cada clase:')
print('=========================')
print(data_df['target'].value_counts(normalize=True))
print()
# Esta desbalanceado (~7% positivos), por eso usamos class_weight='balanced'
# en todos los clasificadores y miramos F1/recall/AUC, no solo accuracy.

# ==========================================
# SELECCIÓN DE FEATURES
# ==========================================
# Justificación de exclusiones:
#   - asteroid_designation / asteroid_fullname / close_approach_date:
#       identificadores o texto libre, sin valor predictivo generalizable.
#   - panic_level / threat_category / panic_verdict:
#       pertenecian al planteamiento anterior.
#   - sentry_impact_prob / sentry_torino_scale / sentry_palermo_scale /
#     sentry_diameter_km:
#       FUGA DE INFORMACIÓN. Solo existen cuando on_sentry_list=True.
#   - H_calculado:
#       correlación 0.99996 con absolute_magnitude -> redundante.
#
# Vectores de posición/velocidad (ast_pre_x/y/z, ast_pre_vx/vy/vz,
# tierra_ca_x/y/z): SE INCLUYEN. Representan la posición y velocidad del
# asteroide y de la Tierra 30 días antes del close approach (calculados
# así explícitamente, de ahí el prefijo "pre"). Por eso la distancia
# euclídea entre ast_pre_* y tierra_ca_* NO coincide con distance_au: son
# instantes distintos (T-30 días vs el momento de máxima aproximación),
# no es una inconsistencia del dataset.
# En el análisis exploratorio, cada componente por separado correlaciona
# muy débilmente con on_sentry_list (todas <0.01 en valor absoluto) y un
# Random Forest con vs sin estas 9 columnas apenas cambia F1/AUC. Se
# incluyen igualmente porque: (a) sirven de base para el estudio
# descriptivo de órbitas del TFG, y (b) un modelo no lineal puede
# explotar combinaciones entre componentes que la correlación simple no
# capta. Se documenta este resultado modesto como hallazgo, no se oculta.
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

columnas_a_mantener = FEATURE_COLUMNS + ['target']
processed_df = data_df[columnas_a_mantener].copy()

# Ya no quedan columnas categóricas en el conjunto de features (se quitó
# threat_category porque deriva del mismo problema que panic_level), así
# que ya no necesitamos el TargetEncoder ni el bloque de categóricas.
# Lo dejamos definido vacío para no romper la estructura del pipeline.
categorical_columns = pd.Index([])
numerical_columns = pd.Index(FEATURE_COLUMNS)

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
# Nota: los vectores ast_pre_*/tierra_ca_* añaden ~4.700 nulos extra
# (asteroides para los que no se pudo calcular la posición a T-30 días),
# por eso la imputación (mean/median) sigue siendo necesaria y se aplica
# por igual a todas las columnas numéricas dentro del pipeline.

# Definir pipeline base
def create_pipeline(scaler, imputer):
    numerical_transformer = Pipeline(steps=[
        ('imputer', imputer),
        ('scaler', scaler)
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns)
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
    scoring='f1',
    n_jobs=-1,
    verbose=1
)


# ==========================================
# SEPARACIÓN TRAIN/TEST (split agrupado por asteroide)
# ==========================================
# IMPORTANTE: un mismo asteroide aparece en varias filas (una por cada
# aproximación a la Tierra en distintas fechas), y on_sentry_list es una
# propiedad del asteroide, no de la fecha concreta. Un train_test_split
# normal (incluso con stratify) dejaría el mismo asteroide repartido entre
# train y test -> data leakage, el modelo memoriza en vez de generalizar.
# GroupShuffleSplit asegura que todas las filas de un mismo asteroide
# caen juntas en train o juntas en test.

X = processed_df.drop(columns=['target'])
y = processed_df['target']
groups = data_df.loc[processed_df.index, 'asteroid_designation']

gss = GroupShuffleSplit(n_splits=1, test_size=1/3, random_state=777)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Verificación de que no hay leakage entre train y test
asteroides_train = set(groups.iloc[train_idx])
asteroides_test = set(groups.iloc[test_idx])
assert len(asteroides_train & asteroides_test) == 0, "¡Hay leakage entre train y test!"
print(f"Asteroides únicos train: {len(asteroides_train)} | test: {len(asteroides_test)}")
print(f"Proporción clase positiva en train: {y_train.mean():.4f} | test: {y_test.mean():.4f}")

# Usamos scoring='f1' en vez de 'accuracy': con ~7% de positivos, un
# modelo que prediga siempre "False" tendría ~93% de accuracy sin haber
# aprendido nada. F1 obliga a acertar también la clase minoritaria.
grid = GridSearchCV(pipeline_base, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
# Entrenamos (aquí es donde se prueban todos los modelos)
grid.fit(X_train, y_train)

# Resultado final
print("******* Los mejores hiperparámetros encontrados son:")
print(grid.best_params_)

mejor_modeloknn = grid.best_estimator_
y_pred = mejor_modeloknn.predict(X_test)

cv_scoresknn = cross_val_score(mejor_modeloknn, X_train, y_train, cv=5, scoring='f1', n_jobs=-1)

# Mostrar los resultados
print("Media de validación cruzada (F1), GridSearchCV:", cv_scoresknn.mean())

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
    scoring='f1',
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
DTC_f1 = f1_score(y_test, y_pred_best)
DTC_precision = precision_score(y_test, y_pred_best)
DTC_recall = recall_score(y_test, y_pred_best)

print(f"Balanced Accuracy: {DTC_Bal_acc:.4f} | Accuracy: {DTC_acc:.4f}")
print(f"F1: {DTC_f1:.4f} | Precision: {DTC_precision:.4f} | Recall: {DTC_recall:.4f}")

# Realizar validación cruzada con 5 folds en los datos de entrenamiento
cv_scoresDTC = cross_val_score(best_DTC, X_train, y_train, cv=5, scoring="f1", n_jobs=-1)

# Mostrar los resultados
print("Media de validación cruzada (F1), hiperparámetros DTC:", cv_scoresDTC.mean())

# Comprobamos cuál es el mejor (comparando F1 medio de validación cruzada,
# la métrica adecuada para un problema desbalanceado; antes se comparaba
# con balanced_accuracy, que es más optimista en este escenario)
if cv_scoresknn.mean() < cv_scoresDTC.mean():
    print("El mejor modelo es el DTC (segunda búsqueda)")
    ElMejorModelo = best_DTC
else:
    print("El mejor modelo es el de la primera búsqueda (LogisticRegression o RandomForest)")
    ElMejorModelo = mejor_modeloknn

# Guardar el modelo entrenado
with open('modelo_sentry_asteroides.pkl', 'wb') as f:
    pickle.dump(ElMejorModelo, f)

print("Modelo guardado como 'modelo_sentry_asteroides.pkl'")

# Reporte de clasificación completo del mejor modelo sobre test
y_pred_final = ElMejorModelo.predict(X_test)
print("\nReporte de clasificación del mejor modelo (test):")
print(classification_report(y_test, y_pred_final, target_names=['No Sentry', 'En Sentry'], zero_division=0))

# AUC-ROC, util como metrica adicional para desbalanceo si el modelo
# expone predict_proba
if hasattr(ElMejorModelo, "predict_proba"):
    y_proba_final = ElMejorModelo.predict_proba(X_test)[:, 1]
    print("AUC-ROC:", roc_auc_score(y_test, y_proba_final))

# ==========================================
# IMPORTANCIA DE VARIABLES (relevante para discutir en la memoria si los
# vectores ast_pre_*/tierra_ca_* aportan al modelo predictivo o solo
# tienen valor para el estudio descriptivo de órbitas)
# ==========================================
clasificador_final = ElMejorModelo.named_steps['classifier']
if hasattr(clasificador_final, 'feature_importances_'):
    importancias = pd.Series(clasificador_final.feature_importances_, index=FEATURE_COLUMNS)
    print("\nImportancia de variables del modelo final:")
    print(importancias.sort_values(ascending=False))