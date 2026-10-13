## MODIFIED Requirements

### Requirement: Champion/challenger con candidatos obligatorios
El sistema SHALL evaluar como mínimo: Seasonal Naive (baseline obligatorio), Pickup Curve basado en `Current Bookings + Expected Remaining Pickup` (baseline obligatorio), ETS, SARIMAX, LightGBM, XGBoost y CatBoost; el campeón resultante SHALL ser el modelo que genera el forecast de la corrida. Cuando el campeón es un modelo ML con rollout recursivo, el forecast SHALL hacer blend por lead: el ML opera los leads 1-30 (donde sus lags son reales y captura el patrón de corto plazo) y la curva de pickup opera los leads 31+ (donde el rollout del ML degrada al realimentar sus propias predicciones como lags).

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
- **WHEN** el campeón de una propiedad es un modelo ML con rollout recursivo (GBM)
- **THEN** las fechas con lead ≤ 30 usan la predicción del GBM (captura el patrón de corto plazo)
- **THEN** las fechas con lead > 30 usan la predicción de la curva de pickup (inmune a la degradación del rollout)
- **THEN** el manifiesto declara qué propiedades operan con blend

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
