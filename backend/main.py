"""
API de predicción de riesgo de mortalidad hospitalaria.

Sirve un modelo entrenado previamente (exportado en modelo_exportado/) detrás
de un único endpoint REST. El formato del modelo (joblib o keras) se resuelve
en runtime leyendo metadata.json, así que si el día de mañana reentrenan y
cambian de sklearn a keras (o viceversa), este archivo no debería tocarse.
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# IMPORTANTE: Se agregan las importaciones necesarias para la clase personalizada
from sklearn.base import BaseEstimator, TransformerMixin


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODELO_DIR = Path("modelo_exportado")
PREPROCESSOR_PATH = MODELO_DIR / "preprocessor.joblib"
METADATA_PATH = MODELO_DIR / "metadata.json"
# El nombre del .keras/.joblib no viene en metadata, así que asumimos la
# convención "modelo.<formato>". Si en algún export usan otro nombre, ajustar acá.


# ---------------------------------------------------------------------------
# Clases personalizadas del modelo (Requeridas por joblib para deserializar)
# ---------------------------------------------------------------------------

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        unidad = X["cod_edad"].astype(str)
        valor = pd.to_numeric(X["edad"], errors="coerce")

        factor = pd.Series(np.nan, index=X.index, dtype="float64")
        factor[unidad.str.contains("Anos") | unidad.str.contains("Años")] = 1.0
        factor[unidad.str.contains("Meses")] = 1.0 / 12.0
        factor[unidad.str.contains("Dias") | unidad.str.contains("Días")] = 1.0 / 365.25
        factor[unidad.str.contains("Horas")] = 1.0 / (24.0 * 365.25)

        X["edad_anios"] = valor * factor
        
        fecha = pd.to_datetime(X["fecha_ingr"], errors="coerce")
        X["dow_ingr"] = fecha.dt.dayofweek.astype("Int64").astype(str)

        X = X.drop(columns=["edad", "cod_edad", "fecha_ingr"])
        return X

import __main__
__main__.FeatureEngineer = FeatureEngineer

# ---------------------------------------------------------------------------
# Contenedor simple para los artefactos cargados en memoria.
# Se guarda en app.state en vez de usar variables globales sueltas, para que
# quede claro qué vive en el ciclo de vida de la app.
# ---------------------------------------------------------------------------

class ModeloRuntime:
    def __init__(self):
        self.preprocessor = None
        self.modelo = None
        self.metadata: dict = {}
        self.formato: str = ""

    def cargar(self):
        if not METADATA_PATH.exists():
            raise FileNotFoundError(f"No se encontró {METADATA_PATH}")

        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.formato = self.metadata["formato_modelo"]

        self.preprocessor = joblib.load(PREPROCESSOR_PATH)

        if self.formato == "joblib":
            self.modelo = joblib.load(MODELO_DIR / "modelo.joblib")
        elif self.formato == "keras":
            # Import diferido: si el modelo es sklearn/joblib no queremos
            # forzar la dependencia de tensorflow ni pagar su tiempo de import.
            from tensorflow import keras
            self.modelo = keras.models.load_model(MODELO_DIR / "modelo.keras")
        else:
            raise ValueError(f"formato_modelo desconocido en metadata.json: {self.formato}")

    def predecir(self, df_row: pd.DataFrame) -> float:
        """Corre el pipeline completo (preprocesador + modelo) y devuelve la proba de clase positiva."""
        X = self.preprocessor.transform(df_row)

        if self.metadata.get("requiere_reshape_features_1"):
            n_features = self.metadata["n_features_tras_preprocesamiento"]
            X = X.reshape(-1, n_features, 1)

        if self.formato == "joblib":
            if hasattr(self.modelo, "predict_proba"):
                proba = self.modelo.predict_proba(X)[:, 1]
            else:
                z = self.modelo.decision_function(X)
                proba = 1 / (1 + np.exp(-z))
        else:
            proba = self.modelo.predict(X).ravel()

        return float(proba[0])


modelo_runtime = ModeloRuntime()


# ---------------------------------------------------------------------------
# Lifespan: carga el preprocesador y el modelo una sola vez, al arrancar.
# Evita el I/O de disco (y el load de keras, que es lento) en cada request.
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        modelo_runtime.cargar()
        print(f"[startup] Modelo '{modelo_runtime.metadata.get('nombre_modelo')}' "
              f"({modelo_runtime.formato}) cargado correctamente.")
    except Exception as e:
        # Si el modelo no carga, preferimos que la app truene al arrancar
        # y no que quede viva respondiendo 500 en cada request.
        raise RuntimeError(f"Error cargando artefactos del modelo: {e}") from e

    yield

    # No hay handles que cerrar explícitamente (joblib/keras no requieren
    # cleanup), pero se deja el bloque por si a futuro se agrega, ej. logging.


app = FastAPI(
    title="API de Riesgo de Mortalidad Hospitalaria",
    description="Expone el modelo entrenado para estimar la probabilidad de fallecimiento de un paciente.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS abierto temporalmente porque se consume desde una GUI externa que
# todavía no tiene dominio fijo. Restringir a los orígenes reales antes de prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PacienteInput(BaseModel):
    """
    Payload crudo de entrada, un registro por paciente.
    Los nombres y el orden respetan columnas_entrada_crudas_esperadas de
    metadata.json; el preprocesador espera exactamente estas columnas.
    """
    prov_ubi: str = Field(..., description="Código de provincia de ubicación del establecimiento")
    cant_ubi: str = Field(..., description="Código de cantón de ubicación del establecimiento")
    area_ubi: str = Field(..., description="Código de área (urbana/rural) de ubicación")
    clase: str = Field(..., description="Clase de establecimiento de salud")
    tipo: str = Field(..., description="Tipo de establecimiento de salud")
    entidad: str = Field(..., description="Entidad/institución responsable")
    sector: str = Field(..., description="Sector (público/privado)")
    nac_pac: str = Field(..., description="Nacionalidad del paciente")
    sexo: str = Field(..., description="Sexo del paciente")
    cod_edad: str = Field(..., description="Código de unidad de la edad (años, meses, días, etc.)")
    edad: int = Field(..., ge=0, description="Edad del paciente en la unidad indicada por cod_edad")
    etnia: str = Field(..., description="Autoidentificación étnica")
    tipo_seg: str = Field(..., description="Tipo de seguro de salud")
    dis_pac: Optional[str] = Field(None, description="Discapacidad del paciente, si aplica")
    prov_res: str = Field(..., description="Código de provincia de residencia del paciente")
    cant_res: str = Field(..., description="Código de cantón de residencia del paciente")
    area_res: str = Field(..., description="Código de área (urbana/rural) de residencia")
    mes_ingr: str = Field(..., description="Mes de ingreso hospitalario")
    dia_ingr: int = Field(..., ge=1, le=31, description="Día de ingreso hospitalario")
    fecha_ingr: str = Field(..., description="Fecha de ingreso, formato YYYY-MM-DD")

    class Config:
        json_schema_extra = {
            "example": {
                "prov_ubi": "17",
                "cant_ubi": "1701",
                "area_ubi": "1",
                "clase": "HOSPITAL",
                "tipo": "GENERAL",
                "entidad": "MSP",
                "sector": "PUBLICO",
                "nac_pac": "ECUATORIANA",
                "sexo": "M",
                "cod_edad": "1",
                "edad": 65,
                "etnia": "MESTIZO",
                "tipo_seg": "NINGUNO",
                "dis_pac": None,
                "prov_res": "17",
                "cant_res": "1701",
                "area_res": "1",
                "mes_ingr": 3,
                "dia_ingr": 14,
                "fecha_ingr": "2024-03-14",
            }
        }


class PrediccionOutput(BaseModel):
    proba: float = Field(..., description="Probabilidad estimada de fallecimiento (0-1)")
    riesgo_alto: bool = Field(..., description="True si proba >= umbral_decision de metadata.json")
    modelo_usado: str = Field(..., description="Nombre del modelo que generó la predicción")


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/api/v1/predict/riesgo-mortalidad", response_model=PrediccionOutput)
async def predecir_riesgo_mortalidad(paciente: PacienteInput):
    """
    Recibe los datos crudos de un paciente, corre el pipeline
    (preprocesador -> modelo) y devuelve la probabilidad de riesgo.
    """
    if modelo_runtime.modelo is None or modelo_runtime.preprocessor is None:
        # Defensivo: no debería pasar si el lifespan cargó bien, pero por si acaso.
        raise HTTPException(status_code=503, detail="El modelo aún no está disponible.")

    # El preprocesador fue entrenado esperando un DataFrame, no un dict/array,
    # así que armamos una fila respetando exactamente las columnas crudas.
    columnas_esperadas = modelo_runtime.metadata["columnas_entrada_crudas_esperadas"]
    fila = {col: getattr(paciente, col) for col in columnas_esperadas}
    df_row = pd.DataFrame([fila], columns=columnas_esperadas)

    try:
        proba = modelo_runtime.predecir(df_row)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar la predicción: {e}") from e

    umbral = modelo_runtime.metadata["umbral_decision"]

    return PrediccionOutput(
        proba=proba,
        riesgo_alto=proba >= umbral,
        modelo_usado=modelo_runtime.metadata.get("nombre_modelo", "desconocido"),
    )


@app.get("/health")
async def health():
    """Chequeo simple para saber si la API levantó y el modelo está en memoria."""
    return {
        "status": "ok",
        "modelo_cargado": modelo_runtime.modelo is not None,
        "modelo": modelo_runtime.metadata.get("nombre_modelo"),
        "formato": modelo_runtime.formato,
    }