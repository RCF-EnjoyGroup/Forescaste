## MODIFIED Requirements

### Requirement: Champion/challenger con candidatos obligatorios
El sistema SHALL evaluar como mínimo: Seasonal Naive (baseline obligatorio), Pickup Curve basado en `Current Bookings + Expected Remaining Pickup` (baseline obligatorio), ETS, SARIMAX, LightGBM, XGBoost y CatBoost; el campeón resultante SHALL ser el modelo que genera el forecast de la corrida.

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
