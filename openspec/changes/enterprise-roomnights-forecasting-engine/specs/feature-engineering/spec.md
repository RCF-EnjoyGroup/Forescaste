## Purpose

Define las features hoteleras del motor — calendario, demanda histórica, capacidad y pickup — calculadas siempre en orden temporal y sin leakage.

## ADDED Requirements

### Requirement: Features de calendario y estacionalidad
El sistema SHALL generar features de calendario: día de semana, fin de semana, semana del año, mes, trimestre, año, día del mes, días hasta fin de mes, indicador de feriado cuando exista información confiable y temporada.

#### Scenario: Horizonte futuro
- **WHEN** se predicen fechas futuras
- **THEN** las features de calendario se derivan de la fecha de estancia sin requerir datos observados

#### Scenario: Sin calendario de feriados
- **WHEN** no existe fuente confiable de feriados
- **THEN** el indicador de feriado se omite y se documenta la ausencia

### Requirement: Features de demanda histórica
El sistema SHALL crear lags 1, 7, 14, 28 y 365 (cuando haya histórico suficiente), medias rolling 7/14/28/56, mediana y desviación estándar rolling, mismo día de la semana histórico y mismo mes histórico.

#### Scenario: Histórico insuficiente para lag 365
- **WHEN** una propiedad tiene menos de un año de historial
- **THEN** las features anuales se omiten para esa propiedad sin romper el pipeline
- **THEN** la omisión queda registrada en el reporte de features

#### Scenario: Mismo día de la semana
- **WHEN** se calcula la demanda histórica del mismo día de la semana
- **THEN** solo se usan semanas anteriores a la fecha de corte

### Requirement: Features de capacidad
El sistema SHALL crear capacidad diaria, capacidad del día anterior, cambio de capacidad, media rolling de capacidad, utilización histórica de capacidad y capacidad restante.

#### Scenario: Capacidad restante
- **WHEN** se computa la capacidad restante para una fecha
- **THEN** se resta de la capacidad la demanda ya reservada conocida a la fecha de corte

### Requirement: Features de pickup
El sistema SHALL crear: RN reservados actuales, lead days, pickup de los últimos 1/3/7/14/30 días, pickup histórico/medio/mediana al mismo lead, volatilidad del pickup, aceleración y desaceleración del pickup, pickup acumulado y pickup como porcentaje de la demanda final.

#### Scenario: Aceleración de pickup
- **WHEN** se calcula la aceleración del pickup
- **THEN** se obtiene como pickup reciente menos pickup del periodo anterior
- **THEN** el signo identifica aceleración o desaceleración de la demanda

#### Scenario: Pickup al mismo lead
- **WHEN** se calcula el pickup histórico para un lead time específico
- **THEN** solo se usan observaciones cuyo snapshot es anterior o igual a la fecha de corte

### Requirement: Curvas de pickup no lineales
El sistema SHALL modelar el pickup como una curva no lineal por lead time, propiedad, día de semana, mes, temporada, cercanía a la fecha de estancia, demanda histórica y nivel actual de reservas, sin asumir linealidad.

#### Scenario: Trayectoria bookings → pickup → final
- **WHEN** se evalúa una fecha con distintos lead times
- **THEN** el comportamiento esperado sigue la trayectoria `Current Bookings → Expected Pickup → Final Room Nights` con pickup decreciente conforme el lead disminuye, salvo evidencia histórica en contrario
