# Predictor de Riesgo de Mortalidad Hospitalaria en Ecuador (INEC 2024)
## Proyecto de Fin de Curso — Inteligencia Artificial (FIEC / ESPOL)

### Integrantes — Grupo #3 (Paralelo #1):
* **Leonardo Zambrano**
* **Javier Murillo**
* **Joel Guamaní**
* **Profesor:** Enrique Peláez J. Ph.D.
* **Fecha de Entrega:** 18 de agosto de 2026

---

## 1. Descripción General del Proyecto

Este sistema es una plataforma de apoyo a la decisión médica para la **estratificación y triaje del riesgo de mortalidad intrahospitalaria** al momento de la admisión del paciente en establecimientos de salud del Ecuador. 

El sistema fue entrenado y evaluado sobre el **censo nacional completo de 1,132,667 egresos hospitalarios** del Instituto Nacional de Estadística y Censos (**INEC 2024**). Para evitar la fuga de información (*Data Leakage*), el modelo utiliza exclusivamente **20 variables disponibles al momento del ingreso**, descartando variables conocidas únicamente al alta médica (como diagnósticos definitivos de egreso o días totales de estancia).

### Modelo Seleccionado para Producción:
* **Regresión Logística Balanceada:** Alcanzó una **Sensibilidad / Recall del 75.23%** (detectando a 3 de cada 4 pacientes en riesgo crítico), un **ROC-AUC de 0.8078**, latencia de inferencia inferior a **5 milisegundos** y alta interpretabilidad para la generación de factores explicativos de riesgo.

---

## 2. Estructura del Repositorio

```text
├── backend/                                # Microservicio API REST en FastAPI
│   ├── main.py                             # Endpoints /predict, /health y schemas Pydantic
│   ├── requirements.txt                    # Dependencias del backend
│   └── modelo_exportado/                   # Artefactos del modelo serializado
│       ├── modelo.joblib                   # Pesos del clasificador ganador (Regresión Logística)
│       ├── preprocessor.joblib             # Pipeline Scikit-Learn ajustado (182 variables)
│       ├── metadata.json                   # Metadatos del modelo, umbral 0.50 y columnas
│       └── ejemplos_demo.json              # 20 casos reales para pruebas del backend
├── data_science/                           # Módulo de experimentación y entrenamiento
│   ├── clasificacion_riesgo_fallecimiento.ipynb  # Notebook oficial de Google Colab / Jupyter
│   ├── matrices_confusion/                 # Gráficos de matrices de confusión y curvas ROC
│   └── ejemplos_demo.json                  # Ejemplos de prueba para validación offline
├── frontend/                               # Interfaz gráfica web interactiva
│   ├── index.html                          # Vista principal: Formulario de triaje y predicción
│   ├── stats.html                          # Vista analítica: Dashboard estadístico y curvas ROC (CU-03)
│   ├── about.html                          # Vista documental: Diccionario clínico de 20 variables
│   ├── app.js                              # Lógica cliente, consumo de API y renderizado de factores
│   ├── style.css                           # Estilos visuales sobrios y reglas de impresión PDF (CU-04)
│   ├── ejemplos_demo.json                  # Casos de prueba para carga interactiva en la UI
│   └── assets/                             # Gráficos oficiales de evaluación experimental
├── diagramas_plantuml/                     # Diagramas de análisis y diseño UML del reporte
│   ├── 01_casos_de_uso.puml                # Casos de uso (CU-01 a CU-04)
│   ├── 02_diagrama_estados.puml            # Estados del paciente y prevención de Data Leakage
│   ├── 03_diagrama_actividades.puml        # Proceso de triaje y predicción
│   ├── 04_diagrama_clases_simplificado.puml # Clases de diseño desacopladas
│   ├── 05_diagrama_componentes_pipeline_simplificado.puml # Arquitectura global del pipeline
│   ├── 06_secuencia_cu01_cu02_inferencia.puml # Secuencia de predicción
│   └── 07_secuencia_cu03_cu04_reportes.puml   # Secuencia de estadísticas y PDF
├── ejemplos_demo.json                      # 20 pacientes reales de prueba en la raíz del proyecto
├── requirements.txt                        # Lista completa de librerías del proyecto
└── README.md                               # Guía técnica y manual de despliegue
```

---

## 3. Conjunto de Ejemplos para Correr el Modelo (`ejemplos_demo.json`)

El proyecto incluye el archivo **`ejemplos_demo.json`** en la raíz y en todas las carpetas clave, conteniendo **20 historias clínicas reales extraídas del censo INEC 2024** que abarcan:
* Pacientes de **alto riesgo** (adultos mayores geriátricos, patologías crónicas, hospitales básicos, sin seguro formal).
* Pacientes de **bajo riesgo** (adultos jóvenes, patologías agudas, cobertura de seguro IESS, hospitales de especialidad).
* Pacientes **neonatales / lactantes** (evaluando unidades de edad en horas, días y meses).
* Casos de **24 provincias** del Ecuador (Guayas, Pichincha, Azuay, Loja, Cotopaxi, Manabí, etc.).

### ¿Cómo probar estos ejemplos?
1. **Desde la Interfaz Web:** En la barra superior, haz clic en **«Seleccionar archivo JSON»** y escoge `ejemplos_demo.json`. Se desplegará una tabla con los 20 pacientes para cargarlos al formulario con un solo clic.
2. **Botón Rápido:** Haz clic en **«Cargar ejemplo (Mujer, 78 años, Loja)»** para llenar el formulario al instante con un caso crítico.

---

## 4. Declaración sobre Librerías No Públicas

> **Declaración Institucional:**  
> En este proyecto **NO se utilizó ninguna librería privada, propietaria ni de pago**.  
> Todas las herramientas, algoritmos y librerías empleadas son de **código abierto (Open-Source)** y de acceso público gratuito a través del repositorio oficial de Python (**PyPI**), instalables mediante el comando estándar `pip install -r requirements.txt`.

---

## 5. Guía de Instalación y Ejecución Paso a Paso

### 1. Prerrequisitos
* Tener instalado **Python 3.10** o superior en el sistema operativo (Linux, macOS o Windows).
* Tener instalado **Git** (opcional para clonación).
* Un navegador web moderno (Google Chrome, Firefox, Microsoft Edge).

---

### 2. Clonar o Descomprimir el Proyecto
Abre una terminal o consola y navega al directorio del proyecto:
```bash
cd RiesgoHosp-Ecuador
```

---

### 3. Crear y Activar un Entorno Virtual (Recomendado)

* **En Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
* **En Windows (CMD o PowerShell):**
  ```cmd
  python -m venv venv
  venv\Scripts\activate
  ```

---

### 4. Instalar las Dependencias
Ejecuta la instalación de todas las librerías oficiales del proyecto:
```bash
pip install -r requirements.txt
```

---

### 5. Iniciar el Servidor Backend (FastAPI)
En la terminal con el entorno virtual activo, dirígete a la carpeta `backend` e inicia el servidor Uvicorn:

```bash
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

El servidor quedará activo en:  
**`http://127.0.0.1:8000`**  
Documentación interactiva Swagger: **`http://127.0.0.1:8000/docs`**

---

### 6. Iniciar la Interfaz de Usuario (Frontend Web)
En **otra ventana de la terminal**, navega a la carpeta `frontend` y lanza un servidor web HTTP ligero:

```bash
cd frontend
python -m http.server 3000
```

---

### 7. Abrir la Aplicación en el Navegador
Abrir el navegador e ingresa a:
* **Predictor de Triaje:** **`http://localhost:3000/index.html`**
* **Estadísticas y Patrones Históricos (CU-03):** **`http://localhost:3000/stats.html`**
* **Documentación y Diccionario Clínico:** **`http://localhost:3000/about.html`**


---

## 6. Prueba Rápida de Inferencia vía Terminal (`cURL`)

Puedes probar la predicción enviando un paciente de prueba directamente a la API:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/predict/riesgo-mortalidad" \
     -H "Content-Type: application/json" \
     -d '{
       "prov_ubi": "Loja",
       "cant_ubi": "Loja",
       "area_ubi": "Urbana",
       "clase": "Hospital general",
       "tipo": "Agudo",
       "entidad": "Privados con fines de lucro",
       "sector": "Privado con fines de lucro",
       "nac_pac": "Ecuatoriano/a",
       "sexo": "Mujer",
       "cod_edad": "Años (1 a 115 años de edad)",
       "edad": 78,
       "etnia": "Mestizo/a",
       "tipo_seg": "Ninguno",
       "dis_pac": "Ninguna",
       "prov_res": "Zamora Chinchipe",
       "cant_res": "Yantzaza",
       "area_res": "Urbana",
       "mes_ingr": "Marzo",
       "dia_ingr": 2,
       "fecha_ingr": "2024-03-02"
     }'
```

**Respuesta esperada (< 5 ms):**
```json
{
  "proba": 0.758,
  "riesgo_alto": true,
  "modelo_usado": "Regresion Logistica",
  "umbral_decision": 0.5
}
```


