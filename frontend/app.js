/* ==========================================================================
   app.js
   Lógica del Predictor de Riesgo de Mortalidad Hospitalaria.
   ========================================================================== */

(function () {
  "use strict";

  const API_URL = "http://127.0.0.1:8000/api/v1/predict/riesgo-mortalidad";

  const FIELD_TYPES = {
    prov_ubi: "string", cant_ubi: "string", area_ubi: "string",
    clase: "string", tipo: "string", entidad: "string", sector: "string",
    nac_pac: "string", sexo: "string", cod_edad: "string", edad: "int",
    etnia: "string", tipo_seg: "string", dis_pac: "string",
    prov_res: "string", cant_res: "string", area_res: "string",
    mes_ingr: "string", dia_ingr: "int", fecha_ingr: "string"
  };

  const SAMPLE_PAYLOAD = {
    prov_ubi: "Loja", cant_ubi: "Loja", area_ubi: "Urbana",
    clase: "Hospital general", tipo: "Agudo", entidad: "Privados con fines de lucro",
    sector: "Privado con fines de lucro", nac_pac: "Ecuatoriano/a", sexo: "Mujer",
    cod_edad: "Años (1 a 115 años de edad)", edad: 30, etnia: "Mestizo/a",
    tipo_seg: "Ninguno", dis_pac: "Ninguna", prov_res: "Zamora Chinchipe",
    cant_res: "Yantzaza", area_res: "Urbana", mes_ingr: "Marzo", dia_ingr: 2, fecha_ingr: "2024-03-02"
  };

  const PROVINCIAS_ECUADOR = [
    "Azuay", "Bolívar", "Cañar", "Carchi", "Chimborazo", "Cotopaxi",
    "El Oro", "Esmeraldas", "Galápagos", "Guayas", "Imbabura", "Loja",
    "Los Ríos", "Manabí", "Morona Santiago", "Napo", "Orellana",
    "Pastaza", "Pichincha", "Santa Elena", "Santo Domingo de los Tsáchilas",
    "Sucumbíos", "Tungurahua", "Zamora Chinchipe"
  ];

  const form = document.getElementById("predict-form");
  const btnLoadSample = document.getElementById("btn-load-sample");
  const btnSubmit = document.getElementById("btn-submit");
  const statusLine = document.getElementById("status-line");
  const resultCard = document.getElementById("result-card");
  const resultProba = document.getElementById("result-proba");
  const riskBadge = document.getElementById("risk-badge");
  const resultModel = document.getElementById("result-model");
  const errorBox = document.getElementById("error-box");
  const errorMessage = document.getElementById("error-message");
  const errorDetailList = document.getElementById("error-detail-list");
  
  // Referencias a los Datalists de provincias
  const provUbiList = document.getElementById("list_prov_ubi");
  const provResList = document.getElementById("list_prov_res");
  
  const fileInput = document.getElementById("file-json");
  const jsonTableSection = document.getElementById("json-table-section");
  const patientsTableHead = document.getElementById("patients-table-head");
  const patientsTableBody = document.getElementById("patients-table-body");
  const btnRandomPatient = document.getElementById("btn-random-patient");
  const randomHint = document.getElementById("random-hint");

  const COLUMNAS_PREFERIDAS = ["edad", "sexo", "prov_ubi", "tipo_seg", "clase", "resultado_real"];

  let pacientesCargados = [];

  // Función para poblar los Datalist
  function poblarDatalistProvincias(datalistEl) {
    PROVINCIAS_ECUADOR.forEach(function (provincia) {
      const option = document.createElement("option");
      option.value = provincia;
      datalistEl.appendChild(option);
    });
  }

  poblarDatalistProvincias(provUbiList);
  poblarDatalistProvincias(provResList);

  function llenarFormulario(datos) {
    Object.keys(FIELD_TYPES).forEach(function (fieldName) {
      if (Object.prototype.hasOwnProperty.call(datos, fieldName)) {
        const input = form.elements[fieldName];
        if (input) {
          input.value = datos[fieldName];
        }
      }
    });
  }

  btnLoadSample.addEventListener("click", function () {
    llenarFormulario(SAMPLE_PAYLOAD);
    ocultarTablaPacientes();
    ocultarHintAleatorio();
    setStatus("idle", "Datos de prueba cargados. Listo para predecir.");
    ocultarResultado();
    ocultarError();
  });

  function setStatus(state, texto) {
    statusLine.dataset.state = state;
    statusLine.textContent = texto;
  }

  function mostrarResultado(data) {
    const proba = typeof data.proba === "number" ? data.proba : Number(data.proba);
    const esRiesgoAlto = Boolean(data.riesgo_alto);

    resultProba.textContent = (proba * 100).toFixed(2) + "%";

    riskBadge.textContent = esRiesgoAlto ? "Riesgo alto" : "Riesgo bajo";
    riskBadge.classList.remove("riesgo-alto", "riesgo-bajo");
    riskBadge.classList.add(esRiesgoAlto ? "riesgo-alto" : "riesgo-bajo");

    resultModel.textContent = data.modelo_usado || "No especificado";
    resultCard.classList.remove("hidden");
  }

  function ocultarResultado() {
    resultCard.classList.add("hidden");
  }

  function mostrarError(mensajePrincipal, detalles) {
    errorMessage.textContent = mensajePrincipal;
    errorDetailList.innerHTML = "";
    if (Array.isArray(detalles)) {
      detalles.forEach(function (item) {
        const campo = Array.isArray(item.loc) ? item.loc.join(" → ") : "campo";
        const li = document.createElement("li");
        li.textContent = campo + ": " + item.msg;
        errorDetailList.appendChild(li);
      });
    }
    errorBox.classList.remove("hidden");
  }

  function ocultarError() {
    errorBox.classList.add("hidden");
    errorMessage.textContent = "";
    errorDetailList.innerHTML = "";
  }

  function mostrarTablaPacientes() { jsonTableSection.classList.remove("hidden"); }
  function ocultarTablaPacientes() { jsonTableSection.classList.add("hidden"); }
  
  function mostrarHintAleatorio(texto) {
    randomHint.textContent = texto;
    randomHint.classList.remove("hidden");
  }
  function ocultarHintAleatorio() {
    randomHint.classList.add("hidden");
    randomHint.textContent = "";
  }

  function obtenerColumnasTabla(pacientes) {
    const primerPaciente = pacientes[0] || {};
    const columnasDisponibles = COLUMNAS_PREFERIDAS.filter(function (col) {
      return Object.prototype.hasOwnProperty.call(primerPaciente, col);
    });
    return columnasDisponibles.length > 0 ? columnasDisponibles : Object.keys(primerPaciente).slice(0, 5);
  }

  function renderTablaPacientes(pacientes) {
    const columnas = obtenerColumnasTabla(pacientes);

    patientsTableHead.innerHTML = "";
    const thIndice = document.createElement("th");
    thIndice.textContent = "#";
    patientsTableHead.appendChild(thIndice);

    columnas.forEach(function (col) {
      const th = document.createElement("th");
      th.textContent = col === "prov_ubi" ? "PROVINCIA" : col === "resultado_real" ? "RESULTADO" : col.replace("_", " ");
      patientsTableHead.appendChild(th);
    });

    patientsTableBody.innerHTML = "";
    pacientes.forEach(function (paciente, index) {
      const tr = document.createElement("tr");
      tr.dataset.index = String(index);

      const tdIndice = document.createElement("td");
      tdIndice.textContent = String(index + 1);
      tr.appendChild(tdIndice);

      columnas.forEach(function (col) {
        const td = document.createElement("td");
        let valor = paciente[col];
        if (col === "resultado_real") valor = formatearResultadoReal(valor);
        td.textContent = (valor === undefined || valor === null) ? "—" : String(valor);
        tr.appendChild(td);
      });

      tr.addEventListener("click", function () {
        seleccionarPaciente(index);
      });

      patientsTableBody.appendChild(tr);
    });
  }

  function marcarFilaSeleccionada(index) {
    Array.from(patientsTableBody.children).forEach(function (fila) {
      fila.classList.remove("selected-row");
    });
    const filaSeleccionada = patientsTableBody.querySelector('tr[data-index="' + index + '"]');
    if (filaSeleccionada) {
      filaSeleccionada.classList.add("selected-row");
    }
  }

  function formatearResultadoReal(valor) {
    if (valor === 0 || valor === "0") return "Vivo";
    if (valor === 1 || valor === "1") return "Fallecido";
    return String(valor);
  }

  function seleccionarPaciente(index) {
    const paciente = pacientesCargados[index];
    if (!paciente) return;

    llenarFormulario(paciente);
    ocultarError();
    ocultarResultado();
    marcarFilaSeleccionada(index);

    const numeroPaciente = index + 1;
    if (Object.prototype.hasOwnProperty.call(paciente, "resultado_real")) {
      mostrarHintAleatorio(
        "Paciente #" + numeroPaciente + " seleccionado — Resultado histórico: " +
        formatearResultadoReal(paciente.resultado_real)
      );
    } else {
      mostrarHintAleatorio("Paciente #" + numeroPaciente + " seleccionado.");
    }
    setStatus("idle", "Datos del paciente #" + numeroPaciente + " cargados. Modifica o envía a predecir.");
  }

  function manejarArchivoJSON(event) {
    const archivo = event.target.files[0];
    if (!archivo) return;
    const lector = new FileReader();

    lector.onload = function (loadEvent) {
      let datos;
      try {
        datos = JSON.parse(loadEvent.target.result);
      } catch (err) {
        setStatus("error", "El archivo no contiene un JSON válido.");
        mostrarError("No se pudo interpretar el archivo: " + err.message, null);
        return;
      }

      ocultarError();
      ocultarResultado();

      if (Array.isArray(datos)) {
        if (datos.length === 0) {
          setStatus("error", "El archivo contiene un array vacío.");
          mostrarError("El archivo JSON no tiene pacientes para mostrar.", null);
          ocultarTablaPacientes();
          return;
        }

        pacientesCargados = datos;
        renderTablaPacientes(pacientesCargados);
        mostrarTablaPacientes();
        ocultarHintAleatorio();
        setStatus("idle", "Se cargaron " + pacientesCargados.length + " pacientes. Selecciona uno en la tabla o usa el botón de azar.");

      } else if (datos && typeof datos === "object") {
        pacientesCargados = [];
        ocultarTablaPacientes();
        ocultarHintAleatorio();
        llenarFormulario(datos);
        setStatus("idle", "Datos del paciente cargados desde archivo JSON.");

      } else {
        setStatus("error", "Formato de archivo no soportado.");
        mostrarError("El JSON debe ser un objeto { } o un array de objetos [ ].", null);
      }
    };

    lector.onerror = function () {
      setStatus("error", "No se pudo leer el archivo.");
      mostrarError("Ocurrió un error al leer el archivo seleccionado.", null);
    };

    lector.readAsText(archivo);
    event.target.value = "";
  }

  fileInput.addEventListener("change", manejarArchivoJSON);

  btnRandomPatient.addEventListener("click", function () {
    if (pacientesCargados.length === 0) return;
    const indiceAleatorio = Math.floor(Math.random() * pacientesCargados.length);
    seleccionarPaciente(indiceAleatorio);
  });

  form.addEventListener("reset", function () {
    ocultarHintAleatorio();
    ocultarError();
    ocultarResultado();
    Array.from(patientsTableBody.children).forEach(function (fila) {
      fila.classList.remove("selected-row");
    });
    setStatus("idle", "Esperando datos...");
  });

  function recolectarPayload() {
    const payload = {};
    Object.keys(FIELD_TYPES).forEach(function (fieldName) {
      const input = form.elements[fieldName];
      const rawValue = input ? input.value : "";
      if (FIELD_TYPES[fieldName] === "int") {
        payload[fieldName] = parseInt(rawValue, 10);
      } else {
        payload[fieldName] = rawValue;
      }
    });
    return payload;
  }

  function validarPayload(payload) {
    const camposInvalidos = [];
    Object.keys(FIELD_TYPES).forEach(function (fieldName) {
      const valor = payload[fieldName];
      const esNumero = FIELD_TYPES[fieldName] === "int";
      if (esNumero && Number.isNaN(valor)) {
        camposInvalidos.push({ loc: ["body", fieldName], msg: "Debe ser un número entero válido" });
      }
      if (!esNumero && (valor === undefined || valor === "")) {
        camposInvalidos.push({ loc: ["body", fieldName], msg: "Este campo es obligatorio" });
      }
    });
    return camposInvalidos;
  }

  form.addEventListener("submit", async function (event) {
    event.preventDefault(); 
    ocultarError();
    ocultarResultado();

    const payload = recolectarPayload();
    const erroresLocales = validarPayload(payload);

    if (erroresLocales.length > 0) {
      setStatus("error", "Hay campos incompletos o inválidos.");
      mostrarError("Revisa los siguientes campos antes de enviar:", erroresLocales);
      return;
    }

    btnSubmit.disabled = true;
    setStatus("loading", "Procesando predicción en el servidor...");

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        let mensaje = "La API respondió con el código " + response.status + ".";
        let detalles = null;
        try {
          const errorData = await response.json();
          if (response.status === 422 && Array.isArray(errorData.detail)) {
            mensaje = "Error de validación (422): revisa los campos indicados.";
            detalles = errorData.detail;
          } else if (typeof errorData.detail === "string") {
            mensaje = errorData.detail;
          }
        } catch (parseErr) {}

        setStatus("error", "Ocurrió un error al procesar la solicitud.");
        mostrarError(mensaje, detalles);
        document.getElementById("result-title").scrollIntoView({ behavior: "smooth", block: "start" });
        return;
      }

      const data = await response.json();
      setStatus("success", "Predicción generada correctamente.");
      mostrarResultado(data);
      
      document.getElementById("result-title").scrollIntoView({ behavior: "smooth", block: "start" });

    } catch (networkError) {
      setStatus("error", "No se pudo conectar con la API.");
      mostrarError("Error de red: verifica que el backend de FastAPI esté corriendo en " + API_URL + ".", null);
      document.getElementById("result-title").scrollIntoView({ behavior: "smooth", block: "start" });
    } finally {
      btnSubmit.disabled = false;
    }
  });

  setStatus("idle", "Esperando datos...");
})();