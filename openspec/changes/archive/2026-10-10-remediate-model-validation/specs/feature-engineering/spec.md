## MODIFIED Requirements

### Requirement: Features de demanda histórica
El sistema SHALL crear lags 1, 7, 14, 28 y 365 (cuando haya histórico suficiente), medias rolling 7/14/28/56, mediana y desviación estándar rolling, mismo día de la semana histórico y mismo mes histórico; todas alineadas a calendario y reportadas con atribución por propiedad.

#### Scenario: Histórico insuficiente para lag 365
- **WHEN** una propiedad tiene menos de un año de historial
- **THEN** las features anuales se omiten para esa propiedad sin romper el pipeline
- **THEN** la omisión queda registrada en el reporte de features

#### Scenario: Mismo día de la semana
- **WHEN** se calcula la demanda histórica del mismo día de la semana
- **THEN** solo se usan semanas anteriores a la fecha de corte

#### Scenario: Lags alineados a calendario
- **WHEN** la serie de una propiedad tiene fechas faltantes
- **THEN** los lags y rollings se calculan sobre la serie reindexada a frecuencia diaria, de modo que `lag_7` corresponde a 7 días naturales atrás y `lag_365` al mismo día del año anterior
- **THEN** ningún hueco desplaza la semántica temporal de una feature

#### Scenario: Atribución de features faltantes
- **WHEN** una feature se omite por histórico insuficiente de una propiedad
- **THEN** el reporte identifica la propiedad o propiedades afectadas
- **THEN** una omisión local nunca se reporta como faltante global de la corrida

#### Scenario: Semántica del mismo mes histórico
- **WHEN** se calcula la feature de mismo mes histórico
- **THEN** su valor corresponde al mismo día (o al más cercano declarado) del año anterior por calendario, distinguible de un desplazamiento posicional de filas

### Requirement: Features de capacidad
El sistema SHALL crear capacidad diaria, capacidad del día anterior, cambio de capacidad, media rolling de capacidad, utilización histórica de capacidad y capacidad restante.

#### Scenario: Capacidad restante
- **WHEN** se computa la capacidad restante para una fecha
- **THEN** se resta de la capacidad la demanda ya reservada conocida a la fecha de corte

#### Scenario: Disponibilidad efectiva de features de capacidad
- **WHEN** la corrida produce el dataset de features
- **THEN** las features de capacidad que dependen de reservas se crean cuando los datos as-of están disponibles en el pipeline
- **THEN** cualquier ausencia no prevista queda registrada en el reporte de features
