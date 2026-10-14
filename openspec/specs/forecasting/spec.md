# forecasting Specification

## Purpose

Define la estrategia de forecasting del motor: candidatos champion/challenger, jerarquía propiedad/portafolio, horizonte configurable, restricción de capacidad, ocupación implícita e intervalos de incertidumbre.

## Requirements

### Requirement: Target y granularidad
El sistema SHALL producir el target `Final Room Nights` a granularidad `Property × Stay Date`, usando exclusivamente `capacity_daily`, `demand_daily` y `pickup_by_lead` como fuentes.

#### Scenario: Salida principal
- **WHEN** se ejecuta el forecast
- **THEN** existe un registro por propiedad y fecha de estancia con las Room Nights esperadas

#### Scenario: Granularidades adicionales
- **WHEN** existen dimensiones consistentes (room type, market segment, channel) en las fuentes
- **THEN** el sistema puede evaluar granularidades adicionales sin crear granularidades artificiales
- **THEN** la granularidad base `Property × Stay Date` nunca se abandona


### Requirement: Demand-first
El sistema SHALL modelar la demanda de Room Nights como variable objetivo; Revenue no será el objetivo de esta versión.

#### Scenario: Objetivo de entrenamiento
- **WHEN** se entrena cualquier candidato
- **THEN** el target es un número de Room Nights, no ingresos


### Requirement: Champion/challenger con candidatos obligatorios
El sistema SHALL evaluar como mínimo: Seasonal Naive (baseline obligatorio), Pickup Curve basado en `Current Bookings + Expected Remaining Pickup` (baseline obligatorio), ETS, SARIMAX, LightGBM, XGBoost y CatBoost; el campeón resultante SHALL ser el modelo que genera el forecast de la corrida. Cuando el campeón es un modelo ML con rollout recursivo y el auto-diagnóstico detecta colapso del rollout a largo lead, el sistema SHALL elegir el reemplazo de los leads largos por evidencia: el candidato con mejor `wmape_long` en los folds de esa propiedad, y solo cuando sea materialmente mejor que el propio campeón; sin esa evidencia el campeón conserva su rollout completo. Cuando el reemplazo elegido es un modelo sin patrón semanal a largo horizonte (SARIMAX/ETS), el segmento empalmado SHALL combinar el nivel del candidato con el perfil DOW de la propiedad calculado as-of de su historia reciente.

#### Scenario: Baselines presentes
- **WHEN** se ejecuta el benchmark de modelos
- **THEN** Seasonal Naive y Pickup Curve están incluidos y sus métricas se reportan
- **THEN** un modelo complejo solo se declara superior si supera consistentemente a los baselines

#### Scenario: Selección de campeón
- **WHEN** termina la evaluación de candidatos
- **THEN** el campeón se elige con evidencia out-of-sample considerando WMAPE, MAPE, Bias, MAE y estabilidad en ventanas de validación
- **THEN** un modelo con menor error pero bias sistemático fuerte no se declara automáticamente campeón

#### Scenario: El campeón opera el forecast
- **WHEN** el benchmark de una corrida selecciona un campeón
- **THEN** el forecast de esa corrida lo genera el modelo campeón (sin modelo de producción fijado en el código)
- **THEN** el manifiesto, la tabla de forecast y el dataset Power BI declaran el mismo campeón que el benchmark
- **THEN** ningún entregable de la misma corrida declara un campeón distinto de otro

#### Scenario: Campeón GBM con blend por lead
- **WHEN** el campeón de una propiedad es un modelo ML con rollout recursivo y el auto-diagnóstico detecta colapso del rollout a largo lead
- **THEN** las fechas con lead ≤ 30 usan la predicción del campeón ML (captura el patrón de corto plazo)
- **THEN** las fechas con lead > 30 usan la predicción del candidato con mejor `wmape_long` en los folds de esa propiedad — que puede ser cualquier candidato del tablero, no un modelo fijado — siempre que sea materialmente mejor que el propio campeón
- **THEN** el manifiesto declara qué propiedades operan con blend, el modelo largo elegido y la evidencia numérica de la elección

#### Scenario: Reemplazo plano con perfil DOW
- **WHEN** el reemplazo largo elegido es un modelo stats cuyo forecast a largo horizonte es plano (sin patrón semanal)
- **THEN** el segmento empalmado combina el NIVEL del candidato con el PERFIL DOW de la propiedad (media por día-de-semana de su historia reciente ≤ corte, normalizada)
- **THEN** el patrón se calcula solo con historia as-of: sin leer reales posteriores al corte
- **THEN** con historia insuficiente para un perfil estable, el segmento queda plano y el manifiesto lo declara

#### Scenario: Colapso del rollout sin reemplazo mejor
- **WHEN** el rollout del campeón ML colapsa Y ningún candidato supera materialmente el `wmape_long` del propio campeón
- **THEN** el campeón conserva su rollout completo en todo el horizonte (el colapso distribucional no es un problema de precisión)
- **THEN** el manifiesto declara el colapso detectado y la decisión de retener al campeón con su evidencia

#### Scenario: Sin evidencia de folds no hay reemplazo
- **WHEN** la corrida no dispone de fold metrics con `wmape_long` por propiedad
- **THEN** el colapso del rollout se registra pero NO se reemplaza ningún tramo del horizonte

#### Scenario: Modelo operativo identificable
- **WHEN** se consume cualquier entregable de forecast
- **THEN** el modelo que lo generó queda identificado por nombre y versión consistentes con el benchmark de la corrida


### Requirement: Selección distinta por propiedad
El sistema SHALL permitir que cada propiedad tenga su propio modelo ganador cuando la evidencia estadística lo justifique, y SHALL operar la selección resultante.

#### Scenario: Modelos diferentes por propiedad
- **WHEN** la validación out-of-sample muestra que propiedad A gana con LightGBM, B con Pickup Curve y C con Seasonal Naive
- **THEN** el sistema selecciona y opera el modelo ganador por propiedad
- **THEN** la selección por propiedad no degrada la métrica agregada del portafolio

#### Scenario: Folds por propiedad
- **WHEN** se evalúa la selección por propiedad
- **THEN** los folds se calculan sobre series por propiedad, no solo sobre la serie agregada del portafolio

#### Scenario: Evidencia insuficiente por propiedad
- **WHEN** una propiedad no acumula folds suficientes para una selección propia fiable
- **THEN** esa propiedad se opera con el campeón global y la fallback queda registrada en el manifiesto


### Requirement: Restricción de capacidad física
El sistema SHALL garantizar que el forecast final nunca supere la capacidad disponible; la capacidad es una restricción operacional del resultado, no solo una variable predictiva.

#### Scenario: Aplicación del techo
- **WHEN** el forecast estadístico/ML supera la capacidad para una fecha
- **THEN** `final_forecast = min(raw_forecast, capacity)`
- **THEN** se conservan por separado `raw_forecast`, `capacity` y `final_forecast`

#### Scenario: Alerta de violación
- **WHEN** `raw_forecast > capacity`
- **THEN** el sistema emite una alerta de constraint violation antes de aplicar el cap


### Requirement: Ocupación implícita
El sistema SHALL calcular `Forecast Occupancy = Forecast RN / Capacity` respetando `0% <= Occupancy <= 100%`.

#### Scenario: Ocupación válida
- **WHEN** se reporta el forecast
- **THEN** la ocupación implícita está dentro del rango 0–100% para cada propiedad y fecha


### Requirement: Jerarquía de forecast
El sistema SHALL generar forecast a nivel propiedad (`Property × Stay Date`) y a nivel portafolio (`All Properties × Stay Date`), reconciliando el portafolio con los forecasts individuales.

#### Scenario: Reconciliación
- **WHEN** existen forecasts por propiedad y un forecast de portafolio
- **THEN** el portafolio se reconcilia con la suma de propiedades
- **THEN** cualquier brecha entre ambos se reporta explícitamente


### Requirement: Horizonte configurable
El sistema SHALL soportar horizontes de 30, 60 y 90 días de forma inicial, configurables por variable (por ejemplo `FORECAST_HORIZON = 90`).

#### Scenario: Cambio de horizonte
- **WHEN** se cambia la configuración de horizonte
- **THEN** el pipeline genera forecasts hasta esa ventana sin modificar código


### Requirement: Intervalos de incertidumbre
El sistema SHALL generar P10, P50 y P90 (o lower/forecast/upper) siempre que el modelo lo permita, siendo el P50 el forecast principal; la banda SHALL anclarse al forecast publicado, calibrarse por banda de lead y respetar el techo de capacidad.

#### Scenario: Publicación de percentiles
- **WHEN** el modelo soporta intervalos
- **THEN** la salida incluye p10, p50 y p90 por propiedad y fecha
- **THEN** el p10 <= p50 <= p90 en todos los registros

#### Scenario: P50 coherente con el forecast publicado
- **WHEN** el forecast final fue restringido por capacidad o transformado después de la predicción
- **THEN** el p50 coincide con el forecast publicado en todos los registros
- **THEN** la banda queda anclada al punto publicado, no a la predicción previa a la restricción

#### Scenario: Banda respeta la capacidad
- **WHEN** el p90 excede la capacidad disponible de una fecha
- **THEN** la banda se restringe a la capacidad y la restricción queda registrada

#### Scenario: Calibración por lead
- **WHEN** se publican bandas para un rango de anticipaciones
- **THEN** la amplitud de la banda refleja el error empírico por banda de lead (la banda de un lead 60-89 no hereda la escala de error de leads 1-30)
- **THEN** el origen de los residuos usados se reporta de forma veraz (backtest real o fuente sintética etiquetada como tal)

#### Scenario: Modelo sin intervalos nativos
- **WHEN** un candidato no puede producir percentiles
- **THEN** el sistema lo indica en la salida y documenta el método alternativo usado o la ausencia de intervalos
