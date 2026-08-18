#!/bin/bash

echo "========================================================"
echo " Iniciando Predictor de Riesgo de Mortalidad Hospitalaria "
echo " FIEC / ESPOL - Grupo #3"
echo "========================================================"

# Navegar a la raíz del proyecto (donde esté este script)
cd "$(dirname "$0")"

# 1. Verificar/Crear entorno virtual
if [ ! -d "venv" ]; then
    echo "-> Creando entorno virtual (venv)..."
    python3 -m venv venv
fi

# 2. Activar entorno virtual
echo "-> Activando entorno virtual..."
source venv/bin/activate

# 3. Instalar dependencias
echo "-> Instalando dependencias..."
pip install -r requirements.txt -q

# 4. Iniciar Backend (FastAPI) en segundo plano (Mostrando Logs)
echo "-> Iniciando Backend (FastAPI) en http://127.0.0.1:8000 ..."
cd backend
# NOTA: Quitamos "> /dev/null 2>&1" para poder ver los errores en la terminal
# Quitamos "--reload" porque en segundo plano puede causar conflictos en Mac
python3 -m uvicorn main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!
cd ..

# 5. Iniciar Frontend (HTTP Server) en segundo plano (Silenciado)
echo "-> Iniciando Frontend (UI) en http://localhost:3000 ..."
cd frontend
python3 -m http.server 3000 > /dev/null 2>&1 &
FRONTEND_PID=$!
cd ..

# 6. Abrir en el navegador web por defecto de macOS
echo "-> Abriendo la aplicación en el navegador..."
sleep 3 # Aumentamos la pausa a 3 segundos para darle tiempo a FastAPI
open "http://localhost:3000/index.html"

echo "========================================================"
echo " ✅ Sistemas en ejecución."
echo " 🛑 Presiona [CTRL+C] en esta terminal para detener todo."
echo "========================================================"

# Capturar CTRL+C para matar ambos procesos limpiamente
trap "echo -e '\nDeteniendo servidores...'; kill $BACKEND_PID $FRONTEND_PID; echo 'Servidores apagados.'; exit 0" SIGINT

# Mantener el script corriendo para escuchar el CTRL+C
wait