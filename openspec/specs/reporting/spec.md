# reporting Specification

## Purpose

Define la salida estructurada del forecast, el dataset listo para Power BI, los KPIs ejecutivos y los entregables del proyecto.

## Requirements

### Requirement: Tabla final de forecast
El sistema SHALL generar una tabla de salida con al menos: `property`, `stay_date`, `forecast_date`, `lead_days`, `current_booked_rn`, `capacity`, `raw_forecast_rn`, `final_forecast_rn`, `p10_forecast`, `p50_forecast`, `p90_forecast`, `model_name`, `model_version` y `forecast_run_timestamp`.

#### Scenario: Columnas obligatorias
- **WHEN** se ejecuta el forecast
- **THEN** la tabla de salida contiene todas las columnas obligatorias con valores poblados

#### Scenario: Columnas extendidas
- **WHEN** la información lo permite
- **THEN** la salida agrega `historical_same_day_rn`, `expected_pickup_rn`, `pickup_rate`, `forecast_occupancy`, `forecast_bias` y `confidence_level`


### Requirement: Dataset listo para Power BI
El sistema SHALL producir un dataset preparado para Power BI que permita analizar forecast (Forecast RN, Actual RN, Variance, Variance %, Occupancy, Capacity), accuracy (MAPE, WMAPE, MAE, RMSE, Bias), pickup (Current RN, Expected Pickup, Pickup %, Lead Time) y modelo (Champion, Challenger, Model version); el desglose de accuracy SHALL incluir la dimensión propiedad y el área de modelo SHALL ser coherente con el benchmark de la corrida.

#### Scenario: Cargas del dashboard
- **WHEN** el dataset se carga en Power BI
- **THEN** cada una de las áreas (forecast, accuracy, pickup, modelo) es analizable con sus campos

#### Scenario: Variance
- **WHEN** existen forecast y Actual para la misma propiedad/fecha
- **THEN** el dataset expone la diferencia absoluta y porcentual entre ambos

#### Scenario: Accuracy por propiedad en Power BI
- **WHEN** la corrida evalúa múltiples propiedades
- **THEN** el dataset Power BI incluye el desglose de accuracy por propiedad además del desglose por banda de lead

#### Scenario: Coherencia del campeón en el dataset
- **WHEN** el área de modelo del dataset Power BI declara el campeón
- **THEN** coincide con el campeón del benchmark y del manifiesto de la misma corrida


### Requirement: KPIs ejecutivos
El sistema SHALL producir como mínimo: Forecast Room Nights, Forecast Occupancy, Forecast Accuracy (WMAPE y MAPE), Forecast Bias, Expected Pickup, Capacity Utilization y Forecast vs Previous Forecast; un KPI sin dato disponible SHALL serializarse como `null`.

#### Scenario: Tablero de KPIs
- **WHEN** se ejecuta una corrida de forecast
- **THEN** los siete KPIs están calculados y disponibles para reporte

#### Scenario: KPI sin dato disponible
- **WHEN** un KPI no tiene datos suficientes para calcularse (por ejemplo, no existe una corrida previa comparable)
- **THEN** se serializa como `null` con el motivo documentado, nunca como `NaN`

#### Scenario: Métricas en escala legible
- **WHEN** el reporte visual muestra WMAPE, sesgo, MAPE, n_obs o KPIs de accuracy
- **THEN** WMAPE y sesgo se muestran como porcentaje (se guardan como fracción), MAPE como porcentaje, n_obs y exclusiones como conteos enteros, y los valores inexistentes (NaN) como "—" coherente con el N/D del panel
- **THEN** una nota de auditoría explica la escala de cada métrica

#### Scenario: Excel de vintages con selección operativa anclada
- **WHEN** el reporte mensual de vintajes reconstruye la proyección de cada corte (día 15)
- **THEN** el mapa de modelos por propiedad sale del registro entre corridas (modo `futuro` con `cutoff <= corte`, la selección operativa vigente a esa fecha — memoria por línea, openspec 2026-11-10-registry-mode-memory)
- **THEN** los cortes previos al motor usan el registro futuro más antiguo disponible y la ausencia de registro cae al snapshot histórico — en ambos casos la fuente queda declarada
- **THEN** la hoja de lectura lleva el sello de versión: fecha de generación, fecha de data usada, regla de selección aplicada y mapa de modelos por propiedad, de modo que dos archivos de fechas distintas sean comparables sin adivinar

#### Scenario: Comparativa real-vs-forecast nunca silenciosa
- **WHEN** la corrida es de verificación y su ventana evaluada tiene reales, pero la unión con el horizonte publicado no produce coincidencias
- **THEN** las tarjetas "Forecast vs Real" y "Dispersión" se renderizan con el MOTIVO explícito de la indisponibilidad (p. ej. "la unión de reales con el horizonte publicado no encontró coincidencias — revisar alineación de claves")
- **THEN** solo cuando genuinamente no hay reales (corrida futura: el horizonte aún no existe) las tarjetas se omiten, que es el comportamiento documentado


### Requirement: Reportes del proyecto
El sistema SHALL producir los entregables: Data Quality Report, EDA Report, Feature Engineering Report, Model Benchmark (Seasonal Naive, Pickup Curve, ETS, SARIMAX, LightGBM, XGBoost, CatBoost), Backtesting Report, Forecast Accuracy Report, Future Forecast Dataset, Model Explainability Report, dataset para Power BI y documentación técnica; cada entregable SHALL derivarse de los cálculos reales de la corrida.

#### Scenario: Benchmark completo
- **WHEN** se ejecuta el Model Benchmark
- **THEN** la tabla de resultados incluye una fila por cada uno de los siete candidatos con MAPE, WMAPE, MAE, RMSE, Bias y estabilidad
- **THEN** el campeón se declara con evidencia out-of-sample
- **WHEN** el benchmark proviene de datos sintéticos de reemplazo
- **THEN** el entregable lo etiqueta explícitamente

#### Scenario: Documentación de limitaciones
- **WHEN** se publica la documentación técnica
- **THEN** las limitaciones del modelo quedan documentadas explícitamente

#### Scenario: Accuracy Report completo
- **WHEN** se publica el Forecast Accuracy Report
- **THEN** incluye las métricas globales, el desglose por propiedad y la cobertura de evaluación (propiedades dentro y fuera de la ventana)

#### Scenario: Explicabilidad derivada del modelo real
- **WHEN** se publica el Model Explainability Report
- **THEN** las importancias se derivan del modelo que generó el forecast de la corrida o del método sustituto que aplique, nunca de valores fijos predefinidos
- **WHEN** el modelo campeón no produce importancias nativas
- **THEN** el reporte declara el método sustituto y su limitación

#### Scenario: Backtesting Report con contenido real
- **WHEN** se produce el Backtesting Report de una corrida productiva
- **THEN** contiene las métricas del backtest ejecutado, no una referencia a las pruebas del framework


### Requirement: Estándares de ingeniería
El sistema SHALL organizar el código de forma modular bajo `src/` (data, validation, features, forecasting, models, evaluation, backtesting, explainability, monitoring, reporting), reproducible, configurable y testeable; la lógica no MAY vivir en un único notebook.

#### Scenario: Notebook como interfaz
- **WHEN** se usa un notebook
- **THEN** este actúa como interfaz de análisis y presentación, delegando la lógica en `src/`

#### Scenario: Re-ejecución con datos actualizados
- **WHEN** llegan datos nuevos en las tres fuentes
- **THEN** el sistema se re-ejecuta con los datos actualizados sin modificar código manualmente


### Requirement: Vista HTML por propiedad
El sistema SHALL incluir en `report.html` el comportamiento por propiedad: forecast de cada propiedad con su banda de incertidumbre, techo de capacidad y reales cuando existan; accuracy por propiedad con su modelo operante — con el WMAPE diario Y el semanal lado a lado, marcando el MAPE diario con advertencia cuando el decil inferior de reales cae por debajo de 10 RN —; y las revisiones del backtest T-90..T-1 desglosadas por propiedad y por banda de lead. Las secciones SHALL ser defensivas ante datos ausentes y jamás romper la corrida.

#### Scenario: Forecast por propiedad
- **WHEN** la corrida produce forecast para varias propiedades
- **THEN** el reporte muestra una gráfica por propiedad con su P50, banda P10/P90 y techo de capacidad
- **THEN** el título de cada panel declara el modelo que opera esa propiedad
- **THEN** los reales se superponen cuando la ventana tiene observados

#### Scenario: Accuracy por propiedad
- **WHEN** la evaluación calcula el desglose por propiedad
- **THEN** el reporte muestra el WMAPE por propiedad y una tabla con propiedad, modelo, WMAPE, MAPE, bias y n_obs
- **THEN** la tabla incluye el WMAPE al grano semanal junto al diario cuando el desglose semanal está disponible
- **THEN** el MAPE diario lleva la advertencia de denominador chico cuando corresponde, con la nota que lo explica

#### Scenario: Revisiones del backtest por propiedad
- **WHEN** el backtest T-90..T-1 produce revisiones
- **THEN** el reporte muestra el WMAPE de las revisiones por propiedad y la convergencia del error por banda de lead (una serie por propiedad)
- **THEN** el `backtesting_report.json` incluye el desglose `by_property_lead_band`

#### Scenario: Datos ausentes
- **WHEN** falta alguno de los desgloses o el backtest está vacío
- **THEN** las secciones correspondientes se omiten con un mensaje y la corrida no falla


### Requirement: Reporte auto-explicado
El reporte HTML SHALL ser legible sin conocimiento previo del motor: cada sección de gráfico SHALL llevar una guía de lectura plegable («¿Cómo leer esta sección?»), un glosario de términos (WMAPE, MAPE, bias, lead days, P10/P50/P90, fold) SHALL estar presente, y un panel de interpretación y validación SHALL traducir las métricas de la corrida a veredictos con criterio explícito. Los veredictos son ayuda de lectura: los gates duros de la corrida viven en los criterios de aceptación del runner y en el gate de calidad de datos.

#### Scenario: Panel de interpretación y validación
- **WHEN** la corrida tiene métricas
- **THEN** el panel muestra WMAPE, bias, cobertura de banda (contra el 80% nominal), convergencia del error (lead 1-2 vs 60-89), significancia del campeón (Diebold-Mariano), cold start, drift YoY y estado del registro entre corridas
- **THEN** con tracking vintage disponible: el forecast publicado de la corrida anterior contra los reales materializados (con la cobertura de la banda publicada y la referencia al CSV de detalle) y la señal de rastreo POR PROPIEDAD de la añada con su criterio
- **THEN** con revisión entre corridas: el delta del forecast sobre estancias comunes aún futuras (|Δ| ≤ 5% estable)
- **THEN** la precisión se juzga por horizonte — corto (≤6d), medio (7-29d), largo (30d+) — contra bandas de la INDUSTRIA configurables (perfil resort/estacional por defecto), con pooling ponderado por volumen entre las bandas de lead del grupo; el WMAPE de ventana declara que mezcla horizontes
- **THEN** cada fila declara el criterio con el que se juzga (p. ej. «|sesgo| < 5% calibrado · < 10% visible · > 10% fuerte») y un veredicto visual
- **THEN** la fila de tracking signal marca a los hoteles cold start (⚠ provisional) y las filas de banda citan la medición de la añada publicada reciente cuando cubre los leads del grupo
- **WHEN** una métrica no es medible en el modo de corrida (p. ej. cobertura en corrida futura)
- **THEN** la fila se presenta como N/D con el motivo, nunca se omite silenciosamente

#### Scenario: Guías de lectura plegables
- **WHEN** se renderiza una sección de gráfico
- **THEN** incluye su guía plegable con qué muestra, cómo interpretarla y qué decisión alimenta

#### Scenario: Cobertura en la tabla de accuracy
- **WHEN** la corrida evaluó cobertura de banda por propiedad
- **THEN** la tabla de accuracy incluye la columna de cobertura con código de color contra el 80% nominal y marca ⚠ a las propiedades cold start (provisionales)

#### Scenario: Regeneración de la vista sin re-correr
- **WHEN** se mejora la presentación del reporte
- **THEN** `scripts/regen_report.py` reconstruye `report.html` de una corrida existente desde sus entregables, sin re-entrenar ni re-correr el pipeline


### Requirement: Reporte mensual de vintages por corte
El sistema SHALL proveer un entregable de negocio con la matriz de vintages mensual: para cada corte del día 15 de cada mes (mientras existan reales a esa fecha), la proyección del mes en curso + los 2 siguientes, contra el real de cada mes proyectado. El mes en curso al corte SHALL proyectarse como cierre de mes (reales 1-15 + pronóstico 16-fin); los modelos SHALL ajustarse as-of el corte (historia ≤ corte, sin fuga); los meses sin reales completos SHALL quedar vacíos (nunca inventados); la metodología y sus salvedades SHALL declararse en el propio archivo.

#### Scenario: Matriz de vintages mensual
- **WHEN** se ejecuta `scripts/monthly_vintage_report.py`
- **THEN** el Excel contiene una fila por (corte × propiedad × mes proyectado) con RN Real y RN Pronosticado, y cada mes proyectado queda cubierto por 1-3 vintages según los cortes disponibles
- **THEN** los cortes posteriores al último real se omiten (nov-15 y dic-15 aún no existen) y los meses proyectados sin reales llevan RN Real vacío
