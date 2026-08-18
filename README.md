# Predictor de Riesgo de Mortalidad Hospitalaria en Ecuador (INEC 2024)

## Proyecto de Fin de Curso — Inteligencia Artificial (FIEC / ESPOL)

### Integrantes — Grupo #3 (Paralelo #1):

- **Leonardo Zambrano**
- **Javier Murillo**
- **Joel Guamaní**
- **Profesor:** Enrique Peláez J. Ph.D.
- **Fecha de Entrega:** 18 de agosto de 2026

---

## 1. Descripción general del proyecto

Este sistema es una plataforma de apoyo a la decisión médica para la **estratificación y triaje del riesgo de mortalidad intrahospitalaria** al momento de la admisión del paciente en establecimientos de salud del Ecuador.

El sistema fue entrenado y evaluado sobre el **censo nacional completo de 1,132,667 egresos hospitalarios** del Instituto Nacional de Estadística y Censos (**INEC 2024**). Para evitar la fuga de información (_Data Leakage_), el modelo utiliza exclusivamente **20 variables disponibles al momento del ingreso**, descartando variables conocidas únicamente al alta médica (como diagnósticos definitivos de egreso o días totales de estancia).

### Modelo Seleccionado para Producción:

- **Regresión Logística Balanceada:** Alcanzó una **Sensibilidad / Recall del 75.23%** (detectando a 3 de cada 4 pacientes en riesgo crítico), un **ROC-AUC de 0.8078**, latencia de inferencia inferior a **5 milisegundos** y alta interpretabilidad para la generación de factores explicativos de riesgo.

---

## 2. Estructura del repositorio

```text
├── backend/                                      # Microservicio API REST en FastAPI
│   ├── main.py                                   # Endpoints /predict, /health y schemas Pydantic
│   ├── requirements.txt                          # Dependencias del backend
│   └── modelo_exportado/                         # Artefactos del modelo serializado
│       ├── modelo.joblib                         # Pesos del clasificador ganador
│       ├── preprocessor.joblib                   # Pipeline de Scikit-Learn ajustado (182 variables)
│       ├── metadata.json                         # Metadatos, umbral 0,50 y columnas
│       └── ejemplos_demo.json                    # 20 casos reales para pruebas del backend
├── data_science/                                 # Módulo de experimentación y entrenamiento
│   ├── clasificacion_riesgo_fallecimiento.ipynb  # Notebook oficial de Google Colab / Jupyter
│   ├── matrices_confusion/                       # Matrices de confusión y curvas ROC
│   └── ejemplos_demo.json                        # Ejemplos de prueba para validación offline
├── frontend/                                     # Interfaz gráfica web interactiva
│   ├── index.html                                # Formulario de triaje y predicción
│   ├── stats.html                                # Dashboard estadístico y curvas ROC (CU-03)
│   ├── about.html                                # Diccionario clínico de las 20 variables
│   ├── app.js                                    # Lógica cliente, consumo de API y factores de riesgo
│   ├── style.css                                 # Estilos y reglas de impresión PDF (CU-04)
│   ├── ejemplos_demo.json                        # Casos de prueba para carga interactiva
│   └── assets/                                   # Gráficos oficiales de evaluación experimental
├── diagramas_plantuml/                           # Diagramas UML de análisis y diseño
│   ├── 01_casos_de_uso.puml                      # Casos de uso CU-01 a CU-04
│   ├── 02_diagrama_estados.puml                 # Estados del paciente y prevención de Data Leakage
│   ├── 03_diagrama_actividades.puml             # Proceso de triaje y predicción
│   ├── 04_diagrama_clases_simplificado.puml     # Clases de diseño desacopladas
│   ├── 05_diagrama_componentes_pipeline_simplificado.puml
│   ├── 06_secuencia_cu01_cu02_inferencia.puml   # Secuencia de predicción
│   └── 07_secuencia_cu03_cu04_reportes.puml     # Secuencia de estadísticas y PDF
├── start_mac.sh                                  # Script de automatización para macOS/Linux
├── ejemplos_demo.json                            # 20 pacientes reales de prueba
├── requirements.txt                              # Dependencias completas del proyecto
└── README.md                                     # Guía técnica y manual de despliegue
```

---

## 3. Conjunto de ejemplos para ejecutar el modelo (`ejemplos_demo.json`)

El proyecto incluye el archivo **`ejemplos_demo.json`** en la raíz y en todas las carpetas clave, conteniendo **20 historias clínicas reales extraídas del censo INEC 2024** que abarcan:

- Pacientes de **alto riesgo** (adultos mayores geriátricos, patologías crónicas, hospitales básicos, sin seguro formal).
- Pacientes de **bajo riesgo** (adultos jóvenes, patologías agudas, cobertura de seguro IESS, hospitales de especialidad).
- Pacientes **neonatales / lactantes** (evaluando unidades de edad en horas, días y meses).
- Casos de **24 provincias** del Ecuador (Guayas, Pichincha, Azuay, Loja, Cotopaxi, Manabí, etc.).

### ¿Cómo probar estos ejemplos?

1. **Desde la Interfaz Web:** En la barra superior, haz clic en **«Seleccionar archivo JSON»** y escoge `ejemplos_demo.json`. Se desplegará una tabla con los 20 pacientes para cargarlos al formulario con un solo clic.
2. **Botón Rápido:** Haz clic en **«Cargar ejemplo (Mujer, 78 años, Loja)»** para llenar el formulario al instante con un caso crítico.

---

## 4. Declaración sobre Librerías No Públicas

> En este proyecto **NO se utilizó ninguna librería privada, propietaria ni de pago**.  
> Todas las herramientas, algoritmos y librerías empleadas son de **código abierto (Open-Source)** y de acceso público gratuito a través del repositorio oficial de Python (**PyPI**), instalables mediante el comando estándar `pip install -r requirements.txt`.

---

## 5. Guía de instalación y ejecución

### Opción A: Ejecución rápida mediante script (macOS/Linux)

Para facilitar la evaluación del proyecto, se incluye un script de automatización que permite levantar todo el entorno con un solo comando.

#### 1. Abrir la terminal

Abre una terminal y navega hasta la carpeta raíz del proyecto:

```bash
cd RiesgoHosp-Ecuador
```

#### 2. Otorgar permisos de ejecución

Este paso solo es necesario la primera vez:

```bash
chmod +x start_mac.sh
```

#### 3. Ejecutar el script

```bash
./start_mac.sh
```

El script se encargará de:

- Crear automáticamente el entorno virtual.
- Instalar las dependencias especificadas.
- Levantar el backend en el puerto `8000`.
- Levantar el frontend en el puerto `3000`.
- Abrir el sistema en el navegador.

Para detener los servicios, presiona `Ctrl+C` en la terminal donde se está ejecutando el script.

---

### Opción B: Ejecución manual paso a paso

Esta alternativa permite levantar los servicios individualmente y resulta útil si se utiliza otro sistema operativo.

### 5.1. Prerrequisitos

- **Python 3.10 o superior**.
- Un navegador web moderno, como Google Chrome, Firefox o Safari.
- `pip`, incluido normalmente con la instalación de Python.

### 5.2. Clonar o descomprimir el proyecto

Abre una terminal y navega hasta el directorio del proyecto:

```bash
cd RiesgoHosp-Ecuador
```

### 5.3. Crear y activar un entorno virtual

#### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

### 5.4. Instalar las dependencias

Con el entorno virtual activado, ejecuta:

```bash
pip install -r requirements.txt
```

### 5.5. Iniciar el servidor backend (FastAPI)

Navega hasta la carpeta `backend`:

```bash
cd backend
```

Inicia el servidor mediante Uvicorn:

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

El servidor quedará disponible en:

```text
http://127.0.0.1:8000
```

La documentación interactiva de Swagger estará disponible en:

```text
http://127.0.0.1:8000/docs
```

### 5.6. Iniciar la interfaz de usuario (frontend)

Abre **otra ventana de terminal** y navega hasta la carpeta `frontend`:

```bash
cd RiesgoHosp-Ecuador/frontend
```

Inicia un servidor HTTP ligero:

```bash
python -m http.server 3000
```

### 5.7. Abrir la aplicación en el navegador

Una vez iniciados ambos servicios, accede a las siguientes páginas:

| Funcionalidad                              | Dirección                          |
| ------------------------------------------ | ---------------------------------- |
| Predictor de triaje                        | `http://localhost:3000/index.html` |
| Estadísticas y patrones históricos (CU-03) | `http://localhost:3000/stats.html` |
| Documentación y diccionario clínico        | `http://localhost:3000/about.html` |

---

## 6. Prueba rápida de inferencia mediante cURL

También es posible probar directamente la API desde la terminal utilizando `curl`.

Ejemplo de una solicitud de predicción:

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
  "modelo_usado": "Regresion Logistica"
}
```

---

## 7. Consideraciones finales

El sistema implementa un flujo completo de **preprocesamiento, inferencia, visualización y explicación del riesgo**, separando los componentes de ciencia de datos, backend y frontend.

La arquitectura está diseñada para garantizar que las variables utilizadas durante la predicción correspondan exclusivamente a información disponible al momento de la admisión, reduciendo así el riesgo de **Data Leakage** durante la inferencia.

El proyecto incluye además diagramas UML, casos de prueba, ejemplos de pacientes, documentación clínica y herramientas de visualización para facilitar su evaluación y comprensión.

> **Uso académico:** este proyecto fue desarrollado con fines académicos y de investigación dentro del curso de Inteligencia Artificial de FIEC / ESPOL. Las predicciones generadas no deben utilizarse como sustituto de la evaluación clínica profesional.
