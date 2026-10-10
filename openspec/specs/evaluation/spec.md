# evaluation Specification

## Purpose

Define la evaluación honesta del motor: validación temporal exclusiva, evaluación multi-step por snapshot, métricas completas con manejo de Actual=0, desgloses por lead time, propiedad y estacionalidad, y la selección de campeón.

## Requirements

### Requirement: Validación temporal exclusiva
El sistema SHALL usar exclusivamente validación temporal (expanding window / rolling origin); el split aleatorio train/test está prohibido. Los folds de selección SHALL distribuirse estacionalmente y no solapar la ventana de métricas publicada.

#### Scenario: Ventanas expansivas
- **WHEN** se configuran las ventanas de validación
- **THEN** cada ventana entrena solo con datos anteriores a ella
- **THEN** la validación simula el comportamiento real de producción

#### Scenario: Cobertura estacional de los folds
- **WHEN** el histórico abarca múltiples ciclos estacionales
- **THEN** los folds de selección cubren tramos distintos del año, incluyendo temporada alta cuando el histórico lo permite
- **THEN** el campeón no se decide con folds concentrados en una sola temporada

#### Scenario: Separación selección/reporte
- **WHEN** se publican métricas de una corrida de verificación
- **THEN** la ventana usada para seleccionar el campeón no solapa la ventana de métricas publicada
- **THEN** ambos períodos quedan declarados en el manifiesto


### Requirement: Evaluación multi-step por snapshot
El sistema SHALL evaluar como en producción: congelar los datos disponibles por snapshot, generar el forecast sin acceso a datos futuros y comparar contra los Room Nights reales posteriormente observados.

#### Scenario: Congelación por snapshot
- **WHEN** se evalúa un snapshot histórico
- **THEN** el forecast se genera solo con la información disponible en ese snapshot
- **THEN** la comparación se hace contra los RN reales de la fecha de estancia


### Requirement: Métricas obligatorias
El sistema SHALL calcular MAPE, WMAPE, MAE (en Room Nights), RMSE y Bias en cada evaluación; WMAPE es la métrica corporativa principal al comparar propiedades con volúmenes diferentes.

#### Scenario: Cálculo de métricas
- **WHEN** termina una evaluación
- **THEN** se reportan `MAPE = mean(abs(Actual - Forecast) / Actual) × 100`, `WMAPE = Σ|Actual - Forecast| / ΣActual`, MAE, RMSE y `Bias = Σ(Forecast - Actual) / ΣActual`

#### Scenario: Dirección del error
- **WHEN** se reporta Bias
- **THEN** se identifica claramente si hay sobre-pronóstico o sub-pronóstico, separado de la precisión


### Requirement: Manejo de Actual = 0 en MAPE
El sistema SHALL excluir del MAPE las observaciones con `Actual = 0`, registrar la cantidad excluida y apoyarse en MAE/WMAPE; nunca sustituir el cero por un valor arbitrario.

#### Scenario: Fechas sin demanda
- **WHEN** existen fechas con `Actual = 0`
- **THEN** esas observaciones no entran al MAPE
- **THEN** el reporte indica cuántas observaciones fueron excluidas
- **THEN** MAE y WMAPE se calculan con el conjunto completo


### Requirement: Precisión por lead time
El sistema SHALL evaluar MAPE, WMAPE y Bias por bandas de lead time: 90+ días, 60–89, 30–59, 14–29, 7–13, 3–6 y 1–2 días.

#### Scenario: Reporte por banda
- **WHEN** termina la evaluación
- **THEN** existe una tabla de MAPE/WMAPE/Bias por banda de lead time
- **THEN** se identifica en qué tramo del booking window el forecast se vuelve confiable


### Requirement: Precisión por propiedad
El sistema SHALL calcular MAPE, WMAPE, MAE, RMSE y Bias por cada propiedad, junto con ranking, peores y mejores propiedades, bias y volatilidad del error.

#### Scenario: Tabla por propiedad
- **WHEN** se evalúa un forecast multi-propiedad
- **THEN** existe una fila por propiedad con las cinco métricas
- **THEN** el ranking de precisión se reporta ordenado


### Requirement: Precisión por estacionalidad
El sistema SHALL evaluar el error por mes, día de la semana, fin de semana vs día laboral y temporadas alta/baja/shoulder, identificando patrones de fallo sistemático.

#### Scenario: Patrón sistemático detectado
- **WHEN** los fines de semana se sub-pronostican de forma consistente
- **THEN** el sistema lo reporta como un insight accionable con la magnitud del sesgo


### Requirement: Benchmarks de precisión
El sistema SHALL reportar MAPE con las referencias: <10% excelente, 10–20% generalmente aceptable, 15–25% razonable en propiedades muy estacionales; tratadas como referencia y no como objetivo rígido.

#### Scenario: Lectura del benchmark
- **WHEN** se publica el MAPE
- **THEN** se muestra en qué rango cae frente a las referencias
- **THEN** se destaca la evolución del error en el tiempo por encima de un número aislado


### Requirement: Integridad del benchmark de candidatos
El sistema SHALL evaluar cada candidato con un protocolo íntegro: features de entrenamiento sin el target de la propia fila, rollout multi-step con features as-of correctas, señal de pickup disponible para los candidatos que la soporten y el mismo techo de capacidad aplicado a todos.

#### Scenario: Features sin target en entrenamiento
- **WHEN** un candidato ML entrena con features rolling
- **THEN** la ventana de cada feature rolling excluye el valor del target de la fila que se predice
- **THEN** una feature que incluye el target de la propia fila es detectada y excluida del entrenamiento

#### Scenario: Rollout multi-step correcto
- **WHEN** un candidato ML predice varios pasos hacia adelante
- **THEN** cada paso usa las features as-of del paso que se predice
- **THEN** el rollout no degrada a un forecast plano por desalineación de filas

#### Scenario: Señal de pickup en candidatos ML
- **WHEN** se entrenan los candidatos ML
- **THEN** disponen de la señal de pickup as-of (RN ya reservados y su evolución) como feature cuando los datos la proveen

#### Scenario: Comparabilidad de candidatos
- **WHEN** el benchmark compara candidatos clásicos y ML
- **THEN** todos se evalúan contra los mismos reales en las mismas ventanas y con el mismo post-procesamiento aplicado en producción

#### Scenario: Candidato pickup representativo
- **WHEN** un candidato basado en curvas de pickup participa del benchmark
- **THEN** su implementación de evaluación es representativa del modelo que operaría en producción (mismo grano y método), sin lo cual el candidato se retira del tablero


### Requirement: Significancia estadística de la selección
El sistema SHALL reportar si la ventaja del campeón sobre el subcampeón es estadísticamente distinguible del ruido.

#### Scenario: Ventaja no significativa
- **WHEN** la diferencia de error entre el campeón y el subcampeón no supera una prueba de significancia sobre los folds
- **THEN** el reporte lo declara como empate estadístico
- **THEN** el desempate usa criterios declarados (parsimonia, interpretabilidad, estabilidad) y queda documentado

#### Scenario: Ventaja significativa
- **WHEN** la diferencia de error supera la prueba de significancia
- **THEN** el reporte declara la ventaja como significativa con la estadística y el valor-p


### Requirement: Cobertura de evaluación reportada
El sistema SHALL reportar qué propiedades y fechas participan efectivamente en cada evaluación y cuáles quedan fuera, con el motivo.

#### Scenario: Propiedades fuera de la ventana
- **WHEN** una propiedad no tiene reales dentro de la ventana evaluada
- **THEN** el reporte la lista como excluida de las métricas con el motivo
- **THEN** el conteo de observaciones usadas refleja solo las propiedades efectivamente evaluadas

#### Scenario: Cobertura visible en KPIs
- **WHEN** se publica una métrica global de la corrida
- **THEN** el número de propiedades evaluadas frente al total del portafolio es visible junto a la métrica
