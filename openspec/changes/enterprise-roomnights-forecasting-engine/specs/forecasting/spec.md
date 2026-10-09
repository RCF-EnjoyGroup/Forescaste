## Purpose

Define la estrategia de forecasting del motor: candidatos champion/challenger, jerarquía propiedad/portafolio, horizonte configurable, restricción de capacidad, ocupación implícita e intervalos de incertidumbre.

## ADDED Requirements

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
El sistema SHALL evaluar como mínimo: Seasonal Naive (baseline obligatorio), Pickup Curve basado en `Current Bookings + Expected Remaining Pickup` (baseline obligatorio), ETS, SARIMAX, LightGBM, XGBoost y CatBoost.

#### Scenario: Baselines presentes
- **WHEN** se ejecuta el benchmark de modelos
- **THEN** Seasonal Naive y Pickup Curve están incluidos y sus métricas se reportan
- **THEN** un modelo complejo solo se declara superior si supera consistentemente a los baselines

#### Scenario: Selección de campeón
- **WHEN** termina la evaluación de candidatos
- **THEN** el campeón se elige con evidencia out-of-sample considerando WMAPE, MAPE, Bias, MAE y estabilidad en ventanas de validación
- **THEN** un modelo con menor error pero bias sistemático fuerte no se declara automáticamente campeón

### Requirement: Selección distinta por propiedad
El sistema SHALL permitir que cada propiedad tenga su propio modelo ganador cuando la evidencia estadística lo justifique.

#### Scenario: Modelos diferentes por propiedad
- **WHEN** la validación out-of-sample muestra que propiedad A gana con LightGBM, B con Pickup Curve y C con Seasonal Naive
- **THEN** el sistema selecciona y opera el modelo ganador por propiedad
- **THEN** la selección por propiedad no degrada la métrica agregada del portafolio

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
El sistema SHALL generar P10, P50 y P90 (o lower/forecast/upper) siempre que el modelo lo permita, siendo el P50 el forecast principal.

#### Scenario: Publicación de percentiles
- **WHEN** el modelo soporta intervalos
- **THEN** la salida incluye p10, p50 y p90 por propiedad y fecha
- **THEN** el p10 <= p50 <= p90 en todos los registros

#### Scenario: Modelo sin intervalos nativos
- **WHEN** un candidato no puede producir percentiles
- **THEN** el sistema lo indica en la salida y documenta el método alternativo usado o la ausencia de intervalos
