"""
Script de entrenamiento completo para el proyecto:
Clasificación de Riesgo de Fallecimiento Hospitalario en Ecuador (INEC 2024).

Entrena y evalúa los 5 modelos de IA:
1. Regresión Logística
2. Perceptrón Simple
3. Multi-Layer Perceptron (MLP - Keras)
4. CNN 1D (Keras)
5. LSTM (Keras)

Exporta el modelo ganador y preprocesador para el Backend (FastAPI).
"""

import gc
import json
import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# ---------------------------------------------------------------------------
# 1. Configuración de Columnas y Fuga de Información
# ---------------------------------------------------------------------------
DATA_PATH = Path("inec_egresos_hospitalarios_2024.csv")
TARGET_COL = "con_egrpa"
FALLECIDO_LABELS = ["Fallecido menos de 48 horas", "Fallecido en 48 horas y más"]

# Variables eliminadas para garantizar CERO fuga de información (post-ingreso)
LEAKAGE_COLS = [
    "dia_estad", "esp_egrpa", "anio_egr", "mes_egr", "dia_egr", "fecha_egr",
    "cau_cie10", "causa3", "cap221rx", "cau221rx", "cau298rx",
]

DROP_COLS = [
    "nom_pais", "cod_pais", "mes_inv", "parr_ubi", "parr_res", "anio_ingr"
]

NUMERIC_FEATURES = ["edad_anios", "dia_ingr"]

CATEGORICAL_FEATURES = [
    "prov_ubi", "cant_ubi", "area_ubi", "clase", "tipo", "entidad", "sector",
    "nac_pac", "sexo", "etnia", "tipo_seg", "dis_pac",
    "prov_res", "cant_res", "area_res",
    "mes_ingr", "dow_ingr",
]

RAW_INPUT_COLS = [
    "prov_ubi", "cant_ubi", "area_ubi", "clase", "tipo", "entidad", "sector",
    "nac_pac", "sexo", "cod_edad", "edad", "etnia", "tipo_seg", "dis_pac",
    "prov_res", "cant_res", "area_res", "mes_ingr", "dia_ingr", "fecha_ingr"
]

# ---------------------------------------------------------------------------
# 2. Transformador de Ingeniería de Características
# ---------------------------------------------------------------------------
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        unidad = X["cod_edad"].astype(str)
        valor = pd.to_numeric(X["edad"], errors="coerce")

        factor = pd.Series(np.nan, index=X.index, dtype="float64")
        factor[unidad.str.contains("Anos|Años", case=False, regex=True)] = 1.0
        factor[unidad.str.contains("Meses", case=False, regex=True)] = 1.0 / 12.0
        factor[unidad.str.contains("Dias|Días", case=False, regex=True)] = 1.0 / 365.25
        factor[unidad.str.contains("Horas", case=False, regex=True)] = 1.0 / (24.0 * 365.25)

        X["edad_anios"] = valor * factor
        fecha = pd.to_datetime(X["fecha_ingr"], errors="coerce")
        X["dow_ingr"] = fecha.dt.dayofweek.astype("Int64").astype(str)

        X = X.drop(columns=["edad", "cod_edad", "fecha_ingr"], errors="ignore")
        return X

# ---------------------------------------------------------------------------
# 3. Carga e Ingesta del Dataset
# ---------------------------------------------------------------------------
print("="*70)
print("1. CARGANDO DATASET INEC EGRESOS HOSPITALARIOS 2024...")
print("="*70)

usecols = RAW_INPUT_COLS + [TARGET_COL]
df = pd.read_csv(DATA_PATH, sep=";", usecols=usecols, encoding="utf-8-sig")
print(f"Total registros cargados: {len(df):,}")

# Construir variable objetivo binaria: 1 = Fallecido, 0 = Vivo
df["target"] = df[TARGET_COL].apply(lambda v: 1 if v in FALLECIDO_LABELS else 0).astype(int)
print(f"Distribución Target: Vivo={np.sum(df['target']==0):,} ({np.mean(df['target']==0)*100:.2f}%) | "
      f"Fallecido={np.sum(df['target']==1):,} ({np.mean(df['target']==1)*100:.2f}%)")

y = df["target"].values
X = df[RAW_INPUT_COLS]

del df
gc.collect()

# ---------------------------------------------------------------------------
# 4. Partición Train / Test Estratificada (80% / 20%)
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("2. PARTICIÓN TRAIN / TEST ESTRATIFICADA (80/20)...")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {len(X_train):,} registros | Mortalidad: {np.mean(y_train)*100:.4f}%")
print(f"Test : {len(X_test):,} registros | Mortalidad: {np.mean(y_test)*100:.4f}%")

# ---------------------------------------------------------------------------
# 5. Pipeline de Preprocesamiento
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("3. CONSTRUYENDO Y AJUSTANDO PIPELINE DE PREPROCESAMIENTO...")
print("="*70)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.005, sparse_output=True, dtype=np.float32)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, NUMERIC_FEATURES),
    ("cat", categorical_transformer, CATEGORICAL_FEATURES),
], sparse_threshold=1.0)

preprocessing_pipeline = Pipeline(steps=[
    ("feature_engineering", FeatureEngineer()),
    ("preprocessor", preprocessor),
])

# Ajustar pipeline con datos de train y transformar
X_train_proc = preprocessing_pipeline.fit_transform(X_train, y_train)
X_test_proc = preprocessing_pipeline.transform(X_test)
n_features = X_train_proc.shape[1]
print(f"Dimensión final de características tras preprocesamiento: {n_features}")

# ---------------------------------------------------------------------------
# 6. Manejo de Desbalance con SMOTE + Barajado Aleatorio
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("4. BALANCEO CON SMOTE (SOLO EN CONJUNTO DE ENTRENAMIENTO)...")
print("="*70)

# Para optimizar tiempo de cómputo en redes neuronales masivas manteniendo alta representatividad,
# entrenamos con un subconjunto remuestreado balanceado
smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.25) # 1:4 ratio para entrenamiento robusto
X_train_res, y_train_res = smote.fit_resample(X_train_proc, y_train)

# Barajar aleatoriamente para distribuir uniformemente las clases
shuffle_idx = np.random.RandomState(RANDOM_STATE).permutation(len(y_train_res))
X_train_res = X_train_res[shuffle_idx]
y_train_res = y_train_res[shuffle_idx]
print(f"Distribución tras balanceo: {np.bincount(y_train_res)}")

# Preparar versiones densas para Keras
# Muestreamos 150,000 registros para el entrenamiento de redes si es muy grande
MAX_DL_SAMPLES = min(200000, len(y_train_res))
dl_idx = np.random.RandomState(RANDOM_STATE).permutation(len(y_train_res))[:MAX_DL_SAMPLES]
X_train_dl_sparse = X_train_res[dl_idx]
y_train_dl = y_train_res[dl_idx]

X_train_dense = X_train_dl_sparse.toarray().astype(np.float32)
X_test_dense = X_test_proc.toarray().astype(np.float32)

print(f"Conjunto denso para Deep Learning: {X_train_dense.shape}")

# ---------------------------------------------------------------------------
# 7. Evaluación de los 5 Modelos
# ---------------------------------------------------------------------------
resultados = []
modelos_guardados = {}

def registrar_evaluacion(nombre, y_true, y_pred, y_proba, tiempo_seg):
    rec = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)

    res = {
        "modelo": nombre,
        "recall": float(rec),
        "precision": float(prec),
        "f1": float(f1),
        "accuracy": float(acc),
        "pr_auc": float(pr_auc),
        "auc_roc": float(auc_roc),
        "tiempo_seg": round(tiempo_seg, 2),
    }
    resultados.append(res)
    print(f"\n[{nombre}] -> Recall: {rec:.4f} | Prec: {prec:.4f} | F1: {f1:.4f} | PR-AUC: {pr_auc:.4f} | ROC-AUC: {auc_roc:.4f} | Acc: {acc:.4f} ({tiempo_seg:.1f}s)")
    print(f"Matriz de Confusión:\n{cm}")
    return res

print("\n" + "="*70)
print("5. ENTRENANDO LOS 5 MODELOS DE INTELIGENCIA ARTIFICIAL...")
print("="*70)

# --- MODELO 1: Regresión Logística ---
print("\n--> Entrenando Modelo 1: Regresión Logística...")
t0 = time.time()
modelo_logreg = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", class_weight="balanced", random_state=RANDOM_STATE)
modelo_logreg.fit(X_train_proc, y_train)
y_proba_lr = modelo_logreg.predict_proba(X_test_proc)[:, 1]
y_pred_lr = (y_proba_lr >= 0.5).astype(int)
t_lr = time.time() - t0
registrar_evaluacion("Regresión Logística", y_test, y_pred_lr, y_proba_lr, t_lr)
modelos_guardados["Regresión Logística"] = {"modelo": modelo_logreg, "formato": "joblib", "reshape": False}

# --- MODELO 2: Perceptrón Simple ---
print("\n--> Entrenando Modelo 2: Perceptrón Simple...")
t0 = time.time()
modelo_perc = Perceptron(alpha=0.0001, eta0=0.1, max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
modelo_perc.fit(X_train_proc, y_train)
# Score continuo con sigmoide sobre decision_function
z = modelo_perc.decision_function(X_test_proc)
y_proba_perc = 1.0 / (1.0 + np.exp(-z))
y_pred_perc = modelo_perc.predict(X_test_proc)
t_perc = time.time() - t0
registrar_evaluacion("Perceptrón Simple", y_test, y_pred_perc, y_proba_perc, t_perc)
modelos_guardados["Perceptrón Simple"] = {"modelo": modelo_perc, "formato": "joblib", "reshape": False}

# --- MODELO 3: Multi-Layer Perceptron (Keras) ---
print("\n--> Entrenando Modelo 3: Multi-Layer Perceptron (MLP Keras)...")
t0 = time.time()
keras.backend.clear_session()
mlp_model = keras.Sequential([
    layers.Input(shape=(n_features,)),
    layers.Dense(64, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.1),
    layers.Dense(1, activation="sigmoid")
])
mlp_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[keras.metrics.AUC(name="auc", curve="ROC"), keras.metrics.Recall(name="recall")]
)
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
mlp_model.fit(
    X_train_dense, y_train_dl,
    validation_split=0.15,
    epochs=12,
    batch_size=512,
    callbacks=[early_stop],
    verbose=1
)
y_proba_mlp = mlp_model.predict(X_test_dense, batch_size=2048, verbose=0).ravel()
y_pred_mlp = (y_proba_mlp >= 0.5).astype(int)
t_mlp = time.time() - t0
registrar_evaluacion("MLP (Keras)", y_test, y_pred_mlp, y_proba_mlp, t_mlp)
modelos_guardados["MLP (Keras)"] = {"modelo": mlp_model, "formato": "keras", "reshape": False}

# --- MODELO 4: Red Neuronal Convolucional 1D (CNN 1D Keras) ---
print("\n--> Entrenando Modelo 4: Red Neuronal Convolucional 1D (CNN 1D)...")
t0 = time.time()
keras.backend.clear_session()
X_train_cnn = X_train_dense.reshape(-1, n_features, 1)
X_test_cnn = X_test_dense.reshape(-1, n_features, 1)

cnn_model = keras.Sequential([
    layers.Input(shape=(n_features, 1)),
    layers.Conv1D(32, kernel_size=3, padding="same", activation="relu"),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.1),
    layers.Conv1D(16, kernel_size=3, padding="same", activation="relu"),
    layers.GlobalAveragePooling1D(),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
cnn_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[keras.metrics.AUC(name="auc", curve="ROC"), keras.metrics.Recall(name="recall")]
)
cnn_model.fit(
    X_train_cnn, y_train_dl,
    validation_split=0.15,
    epochs=10,
    batch_size=512,
    callbacks=[early_stop],
    verbose=1
)
y_proba_cnn = cnn_model.predict(X_test_cnn, batch_size=2048, verbose=0).ravel()
y_pred_cnn = (y_proba_cnn >= 0.5).astype(int)
t_cnn = time.time() - t0
registrar_evaluacion("CNN 1D (Keras)", y_test, y_pred_cnn, y_proba_cnn, t_cnn)
modelos_guardados["CNN 1D (Keras)"] = {"modelo": cnn_model, "formato": "keras", "reshape": True}

# --- MODELO 5: Red Neuronal Recurrente LSTM (Keras) ---
print("\n--> Entrenando Modelo 5: Red Neuronal Recurrente (LSTM Keras)...")
t0 = time.time()
keras.backend.clear_session()
lstm_model = keras.Sequential([
    layers.Input(shape=(n_features, 1)),
    layers.LSTM(32, dropout=0.1),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
lstm_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.002),
    loss="binary_crossentropy",
    metrics=[keras.metrics.AUC(name="auc", curve="ROC"), keras.metrics.Recall(name="recall")]
)
lstm_model.fit(
    X_train_cnn, y_train_dl,
    validation_split=0.15,
    epochs=8,
    batch_size=1024,
    callbacks=[early_stop],
    verbose=1
)
y_proba_lstm = lstm_model.predict(X_test_cnn, batch_size=2048, verbose=0).ravel()
y_pred_lstm = (y_proba_lstm >= 0.5).astype(int)
t_lstm = time.time() - t0
registrar_evaluacion("LSTM (Keras)", y_test, y_pred_lstm, y_proba_lstm, t_lstm)
modelos_guardados["LSTM (Keras)"] = {"modelo": lstm_model, "formato": "keras", "reshape": True}

# ---------------------------------------------------------------------------
# 8. Comparativa y Selección del Ganador
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("6. TABLA COMPARATIVA FINAL DE RENDIMIENTO (CONJUNTO DE PRUEBA 20%)")
print("="*70)
df_res = pd.DataFrame(resultados)
df_res = df_res.sort_values(by=["recall", "f1", "pr_auc", "auc_roc"], ascending=False).reset_index(drop=True)
print(df_res.to_string(index=False))

ganador_row = df_res.iloc[0]
ganador_nombre = ganador_row["modelo"]
print(f"\nMODELO GANADOR SELECCIONADO: {ganador_nombre}")
print(f"Criterio Clínico: Mayor Recall en pacientes fallecidos ({ganador_row['recall']:.4f}) con F1={ganador_row['f1']:.4f}, PR-AUC={ganador_row['pr_auc']:.4f}")

# ---------------------------------------------------------------------------
# 9. Exportación de Artefactos para el Backend FastAPI
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("7. EXPORTANDO ARTEFACTOS PARA EL BACKEND...")
print("="*70)

export_dir = Path("RiesgoHosp-Ecuador/backend/modelo_exportado")
export_dir.mkdir(parents=True, exist_ok=True)

# Guardar Preprocesador
joblib.dump(preprocessing_pipeline, export_dir / "preprocessor.joblib")
print(f"✓ Guardado preprocessor.joblib en {export_dir}")

# Guardar Modelo Ganador
info_ganador = modelos_guardados[ganador_nombre]
if info_ganador["formato"] == "keras":
    info_ganador["modelo"].save(export_dir / "modelo.keras")
    print(f"✓ Guardado modelo.keras en {export_dir}")
else:
    joblib.dump(info_ganador["modelo"], export_dir / "modelo.joblib")
    print(f"✓ Guardado modelo.joblib en {export_dir}")

# Guardar Metadatos
metadata = {
    "nombre_modelo": ganador_nombre,
    "formato_modelo": info_ganador["formato"],
    "requiere_reshape_features_1": bool(info_ganador["reshape"]),
    "n_features_tras_preprocesamiento": int(n_features),
    "umbral_decision": 0.50,
    "metricas_test": {
        "recall": float(ganador_row["recall"]),
        "precision": float(ganador_row["precision"]),
        "f1": float(ganador_row["f1"]),
        "pr_auc": float(ganador_row["pr_auc"]),
        "auc_roc": float(ganador_row["auc_roc"]),
        "accuracy": float(ganador_row["accuracy"]),
    },
    "columnas_entrada_crudas_esperadas": RAW_INPUT_COLS
}

with open(export_dir / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"✓ Guardado metadata.json en {export_dir}")

# Guardar Casos de Prueba (Demo)
X_test_sample = X_test.iloc[:10].to_dict(orient="records")
y_test_sample = [int(v) for v in y_test[:10]]

ejemplos_demo = [
    {"caso": i+1, "esperado_fallecido": bool(y_test_sample[i]), "datos_paciente": X_test_sample[i]}
    for i in range(10)
]

with open(export_dir / "ejemplos_demo.json", "w", encoding="utf-8") as f:
    json.dump(ejemplos_demo, f, indent=2, ensure_ascii=False)
with open("ejemplos_demo.json", "w", encoding="utf-8") as f:
    json.dump(ejemplos_demo, f, indent=2, ensure_ascii=False)
print("✓ Guardado ejemplos_demo.json (10 casos de prueba para la interfaz/API)")

print("\n" + "="*70)
print("¡ENTRENAMIENTO Y EXPORTACIÓN COMPLETADOS CON ÉXITO!")
print("="*70)
