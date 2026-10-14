## MODIFIED Requirements

### Requirement: Champion/challenger con candidatos obligatorios
El sistema SHALL evaluar como mínimo: Seasonal Naive (baseline obligatorio), Pickup Curve basado en `Current Bookings + Expected Remaining Pickup` (baseline obligatorio), ETS, SARIMAX, LightGBM, XGBoost y CatBoost; el campeón resultante SHALL ser el modelo que genera el forecast de la corrida. Cuando el campeón es un modelo ML con rollout recursivo y el auto-diagnóstico detecta colapso del rollout a largo lead, el sistema SHALL elegir el reemplazo de los leads largos por evidencia: el candidato con mejor `wmape_long` en los folds de esa propiedad, y solo cuando sea materialmente mejor que el propio campeón; sin esa evidencia el campeón conserva su rollout completo.

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
