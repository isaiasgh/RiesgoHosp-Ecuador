# Clasificación de Riesgo de Mortalidad Hospitalaria en Ecuador

Este proyecto implementa una solución integral de Inteligencia Artificial para clasificar el riesgo de fallecimiento de pacientes en base a registros históricos de egresos hospitalarios del Ecuador (INEC 2024).

El proyecto consta de un pipeline completo: desde la ingesta y preprocesamiento de datos, el entrenamiento y evaluación de 5 modelos de IA, hasta la disponibilidad del modelo ganador a través de una API RESTful.

## Estructura del Proyecto

```text
/proyecto_final_ia
│
├── README.md                    <-- Este documento
├── /data_science                <-- Entorno de análisis y entrenamiento
│   ├── requirements_ds.txt
│   ├── main.py                  <-- Script para extraer muestra balanceada
│   └── clasificacion.ipynb      <-- Jupyter Notebook con los 5 modelos
│
└── /backend                     <-- Entorno de la API REST
    ├── requirements_api.txt
    ├── main.py                  <-- Servidor FastAPI
    └── /modelo_exportado/       <-- Artefactos generados por el notebook
        ├── preprocessor.joblib
        ├── modelo.keras / .joblib
        └── metadata.json

```

---

## 1. Configuración de Entornos

Para mantener la arquitectura limpia, separamos las dependencias de ciencia de datos de las dependencias de la API. Abre tu terminal en la raíz del proyecto.

### 1.1 Entorno para Data Science

```bash
cd data_science
python3 -m venv env_ds

# Activar en Mac/Linux:
source env_ds/bin/activate
# Activar en Windows:
# env_ds\Scripts\activate

pip install -r requirements_ds.txt

```

### 1.2 Entorno para Backend (API)

Abre otra pestaña en tu terminal:

```bash
cd backend
python3 -m venv env_api

# Activar en Mac/Linux:
source env_api/bin/activate
# Activar en Windows:
# env_api\Scripts\activate

pip install -r requirements_api.txt

```

---

## 2. Muestreo de Datos (Opcional)

Para pruebas rápidas locales sin sobrecargar la memoria RAM, se incluye un script que extrae una muestra balanceada del dataset histórico del INEC.

1. Coloca el archivo original `egresos_hospitalarios_2024.csv` dentro de la carpeta `/data_science`.
2. Con el entorno `env_ds` activado, ejecuta:

```bash
cd data_science
python3 main.py
```

Esto generará el archivo `muestra_para_ia.csv`.

---

## 3. Entrenamiento del Modelo

El entrenamiento evalúa modelos lineales, perceptrones y redes neuronales profundas (CNN 1D, LSTM), comparando métricas como F1-Score y Recall, y exportando automáticamente el ganador.

1. Con el entorno `env_ds` activado, levanta el servidor de notebooks:

```bash
cd data_science
jupyter notebook
```

2. Abre el archivo `clasificacion.ipynb` en tu navegador.
3. Ejecuta las celdas secuencialmente.
4. Al finalizar, el código creará automáticamente la carpeta `/modelo_exportado/` y el archivo `ejemplos_demo.json`.
5. **Mueve la carpeta `/modelo_exportado/` completa a la carpeta `/backend/**`.

---

## 4. Levantamiento de la API

Una vez exportado el modelo, levantamos el microservicio FastAPI para consumirlo.

1. Con el entorno `env_api` activado, ejecuta el servidor:

```bash
cd backend
uvicorn main:app --reload
```

2. La API estará disponible en: `http://localhost:8000`

### Pruebas de la API

- **Health Check:** Visita `http://localhost:8000/health` para verificar que el modelo se cargó en memoria correctamente.
- **Documentación Interactiva (Swagger):** Visita `http://localhost:8000/docs` para ver el esquema de datos y probar el endpoint `/api/v1/predict/riesgo-mortalidad` ingresando un payload desde tu archivo `ejemplos_demo.json`.
