## Purpose

Garantiza que el forecast solo use información que habría estado disponible en la fecha de corte, mediante features as-of seguras y pruebas automáticas que detienen el pipeline si detectan lookahead.

## ADDED Requirements

### Requirement: Disponibilidad as-of por fecha de corte
El sistema SHALL construir features y targets usando únicamente información disponible en la fecha de corte del forecast; para una fecha de estancia determinada nunca podrán usarse Room Nights futuras, pickup futuro, información posterior al corte ni variables derivadas del target futuro.

#### Scenario: Congelación de datos por snapshot
- **WHEN** se genera un forecast para una fecha de corte T
- **THEN** todas las features se calculan con datos con observación ≤ T
- **THEN** el corte efectivo se reporta junto al forecast

#### Scenario: Prohibición de variables futuras
- **WHEN** una feature requiere valores posteriores a T para computarse
- **THEN** esa feature se excluye del entrenamiento y de la predicción

### Requirement: Features rolling e históricas sin lookahead
El sistema SHALL calcular lags, ventanas rolling y estadísticas expansivas exclusivamente hacia atrás en el tiempo.

#### Scenario: Rolling window
- **WHEN** se calcula una media rolling de 28 días para la fecha D
- **THEN** solo se incluyen observaciones anteriores o iguales a D
- **THEN** ningún dato posterior a D participa en el promedio

### Requirement: Split exclusivamente temporal
El sistema SHALL prohibir el split aleatorio train/test en cualquier etapa de entrenamiento o evaluación.

#### Scenario: Split aleatorio detectado
- **WHEN** una configuración de validación mezcla observaciones futuras en entrenamiento respecto a la ventana de evaluación
- **THEN** el pipeline falla antes de entrenar

### Requirement: Pruebas automáticas de leakage
El sistema SHALL ejecutar pruebas automáticas que detecten: RN futuros, pickup futuro, capacidad futura, features calculadas con el dataset completo, rolling con observaciones futuras y splits aleatorios; si detecta leakage, el pipeline MUST fallar.

#### Scenario: Leakage detectado
- **WHEN** una prueba de leakage encuentra una feature con información posterior a la fecha de corte
- **THEN** la corrida termina en error indicando la feature y la evidencia
- **THEN** no se publica ningún forecast de esa corrida

#### Scenario: Sin leakage
- **WHEN** todas las pruebas de leakage pasan
- **THEN** el pipeline continúa y registra el resultado de las pruebas en el reporte de la corrida
