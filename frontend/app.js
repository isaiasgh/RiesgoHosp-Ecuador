/* ==========================================================================
   app.js
   Lógica del Predictor de Riesgo de Mortalidad Hospitalaria.
   Grupo #3 - Inteligencia Artificial (FIEC / ESPOL)
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

  const FIELD_LABELS = {
    prov_ubi: "Provincia del establecimiento",
    cant_ubi: "Cantón del establecimiento",
    area_ubi: "Área del establecimiento",
    clase: "Clase de establecimiento",
    tipo: "Tipo de atención",
    entidad: "Entidad",
    sector: "Sector",
    nac_pac: "Nacionalidad",
    sexo: "Sexo",
    cod_edad: "Código de edad",
    edad: "Edad",
    etnia: "Etnia",
    tipo_seg: "Tipo de seguro",
    dis_pac: "Discapacidad del paciente",
    prov_res: "Provincia de residencia",
    cant_res: "Cantón de residencia",
    area_res: "Área de residencia",
    mes_ingr: "Mes de ingreso",
    dia_ingr: "Día de ingreso",
    fecha_ingr: "Fecha de ingreso"
  };

  const SAMPLE_PAYLOAD = {
    prov_ubi: "Loja", cant_ubi: "Loja", area_ubi: "Urbana",
    clase: "Hospital general", tipo: "Agudo", entidad: "Privados con fines de lucro",
    sector: "Privado con fines de lucro", nac_pac: "Ecuatoriano/a", sexo: "Mujer",
    cod_edad: "Años (1 a 115 años de edad)", edad: 78, etnia: "Mestizo/a",
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
  const errorBox = document.getElementById("error-box");
  const errorMessage = document.getElementById("error-message");
  const errorDetailList = document.getElementById("error-detail-list");
  
  // Elementos de Explicabilidad (CU-02) y Triaje
  const triajeBox = document.getElementById("triaje-box");
  const triajeTitle = document.getElementById("triaje-title");
  const triajeText = document.getElementById("triaje-text");
  const factorsList = document.getElementById("factors-list");
  const btnPrintReport = document.getElementById("btn-print-report");

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
  let ultimoPayloadEvaluado = null;
  let ultimoResultadoEvaluado = null;

  // Función para poblar los Datalist de provincias
  function poblarDatalistProvincias(datalistEl) {
    if (!datalistEl) return;
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
    setStatus("idle", "Datos de prueba cargados (Mujer, 78 años, Loja). Listo para predecir.");
    ocultarResultado();
    ocultarError();
  });

  function setStatus(state, texto) {
    statusLine.dataset.state = state;
    statusLine.textContent = texto;
  }

  // =========================================================================
  // Explicabilidad de Factores de Riesgo (CU-02)
  // =========================================================================
  function renderizarFactoresRiesgo(payload, esRiesgoAlto, proba) {
    factorsList.innerHTML = "";
    const factores = [];

    const edadNum = Number(payload.edad);
    const unidadEdad = String(payload.cod_edad || "");

    // 1. Evaluación de Edad
    if (unidadEdad.includes("Años") || unidadEdad.includes("Anos") || unidadEdad === "1") {
      if (edadNum >= 75) {
        factores.push({
          tipo: "danger",
          icon: "[Alerta]",
          titulo: `Edad Geriátrica Avanzada (${edadNum} años)`,
          desc: "Mayor vulnerabilidad fisiológica basal y disminución de reserva funcional."
        });
      } else if (edadNum >= 60) {
        factores.push({
          tipo: "warning",
          icon: "[Vulnerabilidad]",
          titulo: `Adulto Mayor (${edadNum} años)`,
          desc: "Incremento moderado en la probabilidad de complicaciones intrahospitalarias."
        });
      } else if (edadNum >= 40) {
        factores.push({
          tipo: "warning",
          icon: "[Contexto]",
          titulo: `Adulto de Mediana Edad (${edadNum} años)`,
          desc: "Rango etario de madurez; riesgo basal condicionado principalmente por comorbilidades."
        });
      } else if (edadNum >= 18) {
        factores.push({
          tipo: "success",
          icon: "[Favorable]",
          titulo: `Rango Etario Adulto Joven (${edadNum} años)`,
          desc: "Factor protector: mayor reserva fisiológica y tolerancia basal a intervenciones clínicas."
        });
      } else if (edadNum >= 6) {
        factores.push({
          tipo: "success",
          icon: "[Favorable]",
          titulo: `Edad Pediátrica / Escolar (${edadNum} años)`,
          desc: "Rango etario con baja tasa basal de mortalidad intrahospitalaria general."
        });
      } else {
        factores.push({
          tipo: "warning",
          icon: "[Vulnerabilidad]",
          titulo: `Primera Infancia (${edadNum} años)`,
          desc: "Inmadurez del sistema inmunitario y susceptibilidad a descompensación rápida."
        });
      }
    } else {
      // Meses, Días u Horas (Neonatal / Lactante)
      factores.push({
        tipo: "danger",
        icon: "[Alerta]",
        titulo: `Paciente Neonatal / Lactante (${edadNum} ${unidadEdad})`,
        desc: "Condición de alto cuidado crítico por susceptibilidad fisiológica extrema."
      });
    }

    // 2. Cobertura de Seguro
    const seguro = String(payload.tipo_seg || "").toLowerCase();
    if (seguro.includes("ninguno") || seguro.includes("otro")) {
      factores.push({
        tipo: "warning",
        icon: "[Vulnerabilidad]",
        titulo: "Sin Cobertura de Seguro Formal",
        desc: "Frecuentemente asociado con consultas en fases más avanzadas de la enfermedad."
      });
    } else {
      factores.push({
        tipo: "success",
        icon: "[Favorable]",
        titulo: `Aseguramiento Activo (${payload.tipo_seg})`,
        desc: "Acceso facilitado a red integral de prestaciones e insumos."
      });
    }

    // 3. Complejidad Institucional y Tipo de Atención
    const clase = String(payload.clase || "").toLowerCase();
    const tipo = String(payload.tipo || "").toLowerCase();
    if (tipo.includes("crónico") || tipo.includes("cronico")) {
      factores.push({
        tipo: "danger",
        icon: "[Alerta]",
        titulo: "Atención de Tipo Crónica",
        desc: "Presencia de patologías de base de larga evolución o deterioro progresivo."
      });
    }

    if (clase.includes("básico") || clase.includes("basico") || clase.includes("geriátrico") || clase.includes("geriatrico")) {
      factores.push({
        tipo: "warning",
        icon: "[Contexto]",
        titulo: `Nivel de Atención (${payload.clase})`,
        desc: "Establecimiento de complejidad primaria/básica; recursos de terapia intensiva limitados en la unidad."
      });
    } else if (clase.includes("especialidades") || clase.includes("especialidad")) {
      factores.push({
        tipo: "success",
        icon: "[Favorable]",
        titulo: `Centro de Alta Complejidad (${payload.clase})`,
        desc: "Disponibilidad de subespecialidades médicas, soporte quirúrgico y cuidados intensivos avanzados."
      });
    } else {
      factores.push({
        tipo: "warning",
        icon: "[Contexto]",
        titulo: `Nivel Institucional (${payload.clase})`,
        desc: "Establecimiento con capacidad resolutiva estándar para internación y tratamiento médico-quirúrgico."
      });
    }

    // 4. Ubicación Territorial
    const areaRes = String(payload.area_res || "").toLowerCase();
    const areaUbi = String(payload.area_ubi || "").toLowerCase();
    if (areaRes.includes("rural") || areaUbi.includes("rural")) {
      factores.push({
        tipo: "warning",
        icon: "[Contexto]",
        titulo: "Procedencia o Ubicación Rural",
        desc: "Factores de distancia geográfica y posibles demoras en el acceso oportuno."
      });
    }

    // 5. Discapacidad
    const dis = String(payload.dis_pac || "").toLowerCase();
    if (dis && !dis.includes("ninguna") && !dis.includes("sin información")) {
      factores.push({
        tipo: "warning",
        icon: "[Vulnerabilidad]",
        titulo: `Condición de Discapacidad (${payload.dis_pac})`,
        desc: "Requiere protocolos adaptados y asistencia multidisciplinaria en internación."
      });
    }

    // Renderizar en el DOM
    factores.forEach(function (f) {
      const li = document.createElement("li");
      li.className = "factor-item factor-" + f.tipo;
      li.innerHTML = `
        <span class="factor-tag factor-tag-${f.tipo}">${f.icon}</span>
        <div class="factor-content">
          <strong>${f.titulo}</strong>
          <p>${f.desc}</p>
        </div>
      `;
      factorsList.appendChild(li);
    });

    // Configurar Recomendación de Triaje
    if (esRiesgoAlto) {
      triajeBox.className = "triaje-box triaje-box-alto";
      triajeTitle.textContent = "Alerta Médica: Prioridad I / Cuidados Especiales";
      triajeText.textContent = "Se recomienda monitorización hemodinámica continua, valoración por médico especialista en admisión y priorización de cama en unidad de cuidados intensivos o intermedios.";
    } else {
      triajeBox.className = "triaje-box triaje-box-bajo";
      triajeTitle.textContent = "Evolución Favorable: Prioridad III / Observación Estándar";
      triajeText.textContent = "El perfil multivariable del paciente indica bajo riesgo basal de mortalidad intrahospitalaria. Continuar con plan de internación estándar y seguimiento de rutina.";
    }
  }

  function mostrarResultado(data, payload) {
    ultimoPayloadEvaluado = payload;
    ultimoResultadoEvaluado = data;

    const proba = typeof data.proba === "number" ? data.proba : Number(data.proba);
    const esRiesgoAlto = Boolean(data.riesgo_alto);

    resultProba.textContent = (proba * 100).toFixed(2) + "%";

    riskBadge.textContent = esRiesgoAlto ? "Riesgo Alto (Alerta)" : "Riesgo Bajo (Favorable)";
    riskBadge.classList.remove("riesgo-alto", "riesgo-bajo");
    riskBadge.classList.add(esRiesgoAlto ? "riesgo-alto" : "riesgo-bajo");

    // Explicabilidad y Triaje (CU-02)
    renderizarFactoresRiesgo(payload, esRiesgoAlto, proba);

    resultCard.classList.remove("hidden");
  }

  function ocultarResultado() {
    resultCard.classList.add("hidden");
  }

  function obtenerNombreAmigableCampo(campoClave) {
    return FIELD_LABELS[campoClave] || campoClave;
  }

  function mostrarError(mensajePrincipal, detalles) {
    errorMessage.textContent = mensajePrincipal;
    errorDetailList.innerHTML = "";
    if (Array.isArray(detalles)) {
      detalles.forEach(function (item) {
        let claveCampo = "campo";
        if (Array.isArray(item.loc) && item.loc.length > 0) {
          claveCampo = item.loc[item.loc.length - 1];
        } else if (typeof item.loc === "string") {
          claveCampo = item.loc;
        }
        const nombreVisible = obtenerNombreAmigableCampo(claveCampo);
        
        let textoMensaje = item.msg || "Valor no válido";
        if (textoMensaje.includes("required") || textoMensaje.includes("obligatorio") || textoMensaje.includes("missing")) {
          textoMensaje = "Este campo es obligatorio para la evaluación.";
        } else if (textoMensaje.includes("integer") || textoMensaje.includes("entero")) {
          textoMensaje = "Debe ser un número entero válido.";
        }

        const li = document.createElement("li");
        li.textContent = nombreVisible + ": " + textoMensaje;
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
      const nombreCol = col === "prov_ubi" ? "PROVINCIA" : col === "resultado_real" ? "RESULTADO REAL" : obtenerNombreAmigableCampo(col).toUpperCase();
      th.textContent = nombreCol;
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
        "Paciente #" + numeroPaciente + " cargado — Desenlace histórico registrado por INEC: " +
        formatearResultadoReal(paciente.resultado_real)
      );
    } else {
      mostrarHintAleatorio("Paciente #" + numeroPaciente + " seleccionado.");
    }
    setStatus("idle", "Datos del paciente #" + numeroPaciente + " cargados. Presiona «Predecir riesgo».");
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

      if (Array.isArray(datos)) {
        if (datos.length === 0) {
          setStatus("error", "El arreglo JSON está vacío.");
          mostrarError("El archivo contiene una lista sin registros.", null);
          return;
        }
        pacientesCargados = datos;
        renderTablaPacientes(pacientesCargados);
        mostrarTablaPacientes();
        seleccionarPaciente(0);
        setStatus("idle", "Se cargaron " + datos.length + " pacientes de prueba. Selecciona uno para predecir.");
      } else if (typeof datos === "object" && datos !== null) {
        pacientesCargados = [datos];
        ocultarTablaPacientes();
        llenarFormulario(datos);
        ocultarHintAleatorio();
        setStatus("idle", "Paciente cargado desde archivo JSON. Listo para predecir.");
      } else {
        setStatus("error", "Estructura JSON no reconocida.");
        mostrarError("El archivo debe contener un objeto o una lista de pacientes.", null);
      }
    };

    lector.onerror = function () {
      setStatus("error", "Error al leer el archivo.");
      mostrarError("No se pudo leer el archivo seleccionado.", null);
    };

    lector.readAsText(archivo, "UTF-8");
  }

  fileInput.addEventListener("change", manejarArchivoJSON);

  btnRandomPatient.addEventListener("click", function () {
    if (!pacientesCargados || pacientesCargados.length === 0) return;
    const indiceAleatorio = Math.floor(Math.random() * pacientesCargados.length);
    seleccionarPaciente(indiceAleatorio);
  });

  function recolectarPayload() {
    const payload = {};
    const formData = new FormData(form);
    Object.keys(FIELD_TYPES).forEach(function (fieldName) {
      const tipo = FIELD_TYPES[fieldName];
      const valorCrudo = formData.get(fieldName);
      if (tipo === "int") {
        payload[fieldName] = valorCrudo !== null && valorCrudo !== "" ? parseInt(valorCrudo, 10) : NaN;
      } else {
        payload[fieldName] = valorCrudo !== null ? String(valorCrudo).trim() : "";
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
        camposInvalidos.push({ loc: ["body", fieldName], msg: "Debe ser un número entero válido." });
      }
      if (!esNumero && (valor === undefined || valor === "")) {
        camposInvalidos.push({ loc: ["body", fieldName], msg: "Este campo es obligatorio para la evaluación." });
      }
    });
    return camposInvalidos;
  }

  // =========================================================================
  // Envío del Formulario (Inferencia Asíncrona)
  // =========================================================================
  form.addEventListener("submit", async function (event) {
    event.preventDefault(); 
    ocultarError();
    ocultarResultado();

    const payload = recolectarPayload();
    const erroresLocales = validarPayload(payload);

    if (erroresLocales.length > 0) {
      setStatus("error", "Hay campos incompletos en el formulario.");
      mostrarError("Por favor completa los siguientes campos antes de solicitar la predicción:", erroresLocales);
      return;
    }

    btnSubmit.disabled = true;
    setStatus("loading", "Procesando inferencia con el modelo de Inteligencia Artificial...");

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        let mensaje = "No se pudo procesar la solicitud.";
        let detalles = null;
        try {
          const errorData = await response.json();
          if (response.status === 422 && Array.isArray(errorData.detail)) {
            mensaje = "Por favor verifica los siguientes campos del formulario:";
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
      setStatus("success", "Predicción generada con éxito.");
      mostrarResultado(data, payload);
      
      document.getElementById("result-title").scrollIntoView({ behavior: "smooth", block: "start" });

    } catch (networkError) {
      setStatus("error", "No se pudo conectar con el servidor.");
      mostrarError("Error de conexión: verifica que el servidor esté activo en " + API_URL + ".", null);
      document.getElementById("result-title").scrollIntoView({ behavior: "smooth", block: "start" });
    } finally {
      btnSubmit.disabled = false;
    }
  });

  // =========================================================================
  // Botón Descargar / Imprimir Reporte (CU-04)
  // =========================================================================
  if (btnPrintReport) {
    btnPrintReport.addEventListener("click", function () {
      if (!ultimoPayloadEvaluado || !ultimoResultadoEvaluado) {
        alert("Primero genera una predicción para poder exportar el reporte.");
        return;
      }
      window.print();
    });
  }

  setStatus("idle", "Esperando datos...");
})();