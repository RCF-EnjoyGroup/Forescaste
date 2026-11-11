# monitoring Specification

## Purpose

Define el monitoreo operacional del motor: alertas de drift, retraining configurable, almacenamiento de snapshots de forecast y reproducibilidad total de cada corrida.

## Requirements

### Requirement: Detección de drift
El sistema SHALL monitorear deterioro de MAPE y WMAPE, drift de Bias, drift de features, drift de la curva de pickup y drift de la distribución de demanda; el monitoreo SHALL ejecutarse como parte de la corrida productiva y sus resultados quedar registrados en el manifiesto.

#### Scenario: Drift de métricas
- **WHEN** la precisión o el bias se deterioran significativamente respecto a la referencia histórica
- **THEN** el sistema genera una alerta identificando la métrica y la magnitud del deterioro

#### Scenario: Drift de curva de pickup
- **WHEN** la curva de pickup por lead se desplaza respecto de la histórica
- **THEN** el sistema reporta el cambio para que sea revisado antes del próximo retraining

#### Scenario: Monitoreo en la corrida productiva
- **WHEN** termina una corrida productiva end-to-end
- **THEN** la detección de outliers y la suite de drift se ejecutaron sobre los datos de la corrida
- **THEN** sus resultados y la decisión de re-entrenamiento quedan registrados en el manifiesto

#### Scenario: Drift de demanda estacionalmente comparable
- **WHEN** se evalúa el drift de la distribución de demanda de una propiedad
- **THEN** la referencia es la misma ventana calendario un año atrás (YoY) por propiedad, medida con PSI
- **THEN** una propiedad con menos de 365 días de histórico queda documentada como no evaluable (sin alerta inventada)
- **THEN** una alerta que dispararía en cada corrida por crecimiento estructural antiguo del portafolio NO se genera

#### Scenario: Outliers con baseline local
- **WHEN** se calcula el z-score de un día de una propiedad
- **THEN** el baseline es una ventana rodante centrada en la propia fecha (no la media de todo el histórico)
- **THEN** el arranque de baja demanda de una propiedad nueva no se clasifica como outlier


### Requirement: Estado entre corridas (registro)
El sistema SHALL persistir un registro append-only entre corridas con: métricas, fuente de métricas, campeón global y por propiedad, cobertura de banda y timestamp de cada corrida; el registro SHALL escribirse atómicamente y un registro ausente o corrupto SHALL jamás romper la corrida.

#### Scenario: Drift de métricas con memoria
- **WHEN** existe una corrida anterior registrada con fuente de métricas comparable
- **THEN** el drift de MAPE/WMAPE/Bias se compara contra esa corrida (no contra un placeholder vacío)

#### Scenario: Re-entrenamiento con estado real
- **WHEN** se decide si re-entrenar
- **THEN** los días desde el último entrenamiento y los registros nuevos se calculan del registro (no hardcodeados)

#### Scenario: Histéresis del campeón por propiedad
- **WHEN** la selección por propiedad de esta corrida encuentra un ganador local que no supera materialmente al incumbente de la corrida anterior
- **THEN** el incumbente se retiene y el motivo queda documentado

#### Scenario: Tracking vintage del forecast publicado
- **WHEN** la corrida anterior publicó un forecast operativo y parte de su horizonte ya tiene reales
- **THEN** ese forecast se puntúa por vintage (métricas, desglose por lead y cobertura de banda)
- **THEN** la añada graduada es la publicación operativa MÁS RECIENTE cuya ventana ya tiene reales — una futura recién publicada con corte en el último real aún no es graduable (su horizonte entero es futuro); el reporte declara qué añada se graduó (`published_run_id`)
- **THEN** la revisión entre corridas (delta sobre estancias comunes) se publica en los KPIs
- **THEN** el detalle forecast-vs-real por estancia (publicado P50/banda, real, error en RN, error % y si el real cayó dentro de la banda) se publica en `vintage_tracking_detail.csv`, con la corrida que lo publicó (`published_run_id`) para trazabilidad
- **THEN** la señal de rastreo (Σ error con signo ÷ desviación media) se calcula **por propiedad** sobre la añada — el control de sesgo sistemático acumulado que el MAPE esconde, con umbral configurable (clásico |TS| ≤ 4)
- **THEN** el EXPEDIENTE de añadas graduadas se reconstruye desde el registro (deduplicado por publicación — atribución por `published_run_id` o por corte = inicio de la ventana − 1d —, cada añada con su graduación más reciente) y se publica como sección del reporte con tabla y gráfica por añada, más una fila en el panel con la tendencia del expediente y la salvedad explícita del n de cada añada
- **WHEN** ninguna estancia del forecast anterior se ha materializado
- **THEN** el CSV no se publica (nunca se inventa) y el manifiesto lo documenta

#### Scenario: Corrección de nivel por drift (knob, default OFF)
- **WHEN** `monitoring.level_correction.enabled` está activo y la corrida es `futuro` con una publicación previa graduable
- **THEN** el desvío de nivel por propiedad se estima SOLO de los errores del forecast publicado previo sobre la ventana reciente de estancias (`rel = Σ(fc−act)/Σ|act|`, `ts = Σerr/media|err|`), con peso suave `min(|ts|/ts_trigger, 1)`, shrinkage `n/(n+shrinkage)`, clip `±max_rel`, `min_obs` y umbral de materialidad (0.5%)
- **THEN** la aplicación es multiplicativa al punto ANTES de la corrección por banda y de las restricciones; el dataset publicado lleva `level_correction_factor` y el manifiesto el detalle por propiedad (n, rel, ts, peso, factor, razón) — ninguna corrección silenciosa
- **THEN** en modo verificación NO se aplica (el horizonte con reales no se corrige con información del propio horizonte) y la razón queda en el manifiesto
- **THEN** con `enabled: false` el sistema no escribe la columna ni cambia ningún número
- **THEN** el default OFF quedó fijado por el A/B replay del 2026-11-05: la señal (errores de la publicación previa) solo observa leads cortos ya calibrados por el piso de reservas — cuando disparó, sobre-corrigió (cobertura FIESTA 90.8%→81.7%) y el drift real (leads 14-59) queda fuera de su alcance temporal; la maquinaria queda como knob a la espera de la variante de señal de publicación antigua


#### Scenario: Memoria por línea de corrida (modo)
- **WHEN** una corrida pide su estado previo al registro (`previous_run`)
- **THEN** lo toma del último registro del MISMO modo (`real` para verificación, `futuro` para publicación operativa, `ad hoc` para corridas sueltas)
- **THEN** la histéresis del campeón, la memoria de drift de métricas y el estado de re-entrenamiento consumen ese registro de su propia línea — jamás el de la otra
- **THEN** si no existe registro previo de ese modo, la memoria arranca vacía y el manifiesto lo declara — nunca herencia cruzada silenciosa entre líneas
- **THEN** los consumidores de la línea operativa (`previous_operational`, `select_gradable_publication` — que ya filtraban `mode="futuro"`) quedan como referencia del patrón, sin cambio de comportamiento
- **THEN** la regla quedó validada con el replay del registro real (2026-11-10): 24 de 36 corridas cargaban memoria de la otra línea; el caso que motivó el cambio (`real-20261105-1442`, brazo B del A/B de bias) heredaba `sarimax` de una corrida futura y con la regla hereda la última real (`lightgbm`), dejando los brazos del A/B comparables en modelo

#### Scenario: Memoria por línea de corrida (modo) — acceso robusto
- **WHEN** cualquier consumidor lee la ruta del registro (`monitoring.registry_path`)
- **THEN** la lee como mapa (`(cfg.monitoring or {}).get("registry_path")`) — jamás por atributo, que revienta con `AttributeError` contra el `dict` de `src/config.py`
- **THEN** la selección as-of del modo futuro (`run_real --futuro`) encadena al último registro **futuro** con el mismo helper que el runner — un solo camino de lectura del registro para toda la memoria entre corridas
- **THEN** la regla quedó fijada tras el crash del 2026-11-10 (`--futuro` caía con `AttributeError: 'dict' object has no attribute 'registry_path'` desde el commit que añadió la histéresis del modo futuro); el helper `_incumbente_previo` concentra la lectura y los tests de regresión la clavan

### Requirement: Detección de outliers sin eliminación automática
El sistema SHALL detectar eventos, picos anormales, caídas abruptas, cierres, cambios de inventario, datos faltantes y cambios estructurales, clasificándolos sin eliminarlos automáticamente.

#### Scenario: Clasificación de picos
- **WHEN** se detecta un pico anormal de demanda
- **THEN** el sistema lo clasifica (error de datos, evento real, grupo, temporada, cambio estructural)
- **THEN** los eventos reales se conservan en el entrenamiento


### Requirement: Retraining configurable
El sistema SHALL soportar retraining con frecuencia configurable (`RETRAIN_FREQUENCY`), horizonte configurable (`FORECAST_HORIZON`) e histórico mínimo configurable (`MIN_HISTORY`), sin re-entrenar innecesariamente cuando no hay información nueva suficiente.

#### Scenario: Sin información nueva suficiente
- **WHEN** los datos nuevos no alcanzan el mínimo configurado
- **THEN** el sistema omite el retraining y registra el motivo

#### Scenario: Retraining programado
- **WHEN** se cumple la frecuencia configurada con datos nuevos suficientes
- **THEN** el pipeline re-entrena y valida que el desempeño no se haya degradado


### Requirement: Reproducibilidad de cada corrida
El sistema SHALL guardar por cada ejecución: versión del dataset, versión del modelo, periodo de entrenamiento, periodo de validación, configuración de features, hiperparámetros, métricas, timestamp y random seed.

#### Scenario: Corrida reproducible
- **WHEN** se solicita reproducir una corrida previa
- **THEN** con los metadatos almacenados se regeneran los mismos resultados
- **THEN** el forecast es determinístico dada la misma configuración y seed
