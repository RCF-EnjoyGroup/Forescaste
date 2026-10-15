# feature-engineering Specification

## Purpose

Define las features hoteleras del motor — calendario, demanda histórica, capacidad y pickup — calculadas siempre en orden temporal y sin leakage.

## Requirements

### Requirement: Features de calendario y estacionalidad
El sistema SHALL generar features de calendario: día de semana, fin de semana, semana del año, mes, trimestre, año, día del mes, días hasta fin de mes y temporada; el indicador de feriado SHALL activarse cuando exista una fuente confiable configurada (países y ventana de viaje), computando fechas movibles correctamente, y su ausencia SHALL documentarse cuando la fuente esté deshabilitada. Los eventos privados de la fuente opcional SHALL entrar a la feature de distancia al feriado/evento de los candidatos ML y al mapa de ventanas con scoping por propiedad; el indicador `is_holiday` queda reservado a feriados del calendario público.

#### Scenario: Horizonte futuro
- **WHEN** se predicen fechas futuras
- **THEN** las features de calendario se derivan de la fecha de estancia sin requerir datos observados

#### Scenario: Fuente confiable de feriados configurada
- **WHEN** la configuración habilita países de feriados (p. ej. CR + US)
- **THEN** el indicador de feriado se genera para toda fecha de estancia, incluyendo feriados de fecha movible (Semana Santa, Thanksgiving, Memorial Day)
- **THEN** las fechas de feriado son conocidas por adelantado y usables a cualquier lead sin leakage

#### Scenario: Eventos privados en la feature de distancia
- **WHEN** la fuente opcional de eventos está activa
- **THEN** la distancia de cada estancia al feriado/evento más cercano considera los eventos además de los feriados
- **THEN** `is_holiday` solo marca feriados del calendario público, nunca eventos

#### Scenario: Sin calendario de feriados
- **WHEN** la fuente de feriados está deshabilitada por configuración
- **THEN** el indicador de feriado se omite y la ausencia se documenta


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
El sistema SHALL modelar el pickup como una curva no lineal por lead time, propiedad, día de la semana, mes, temporada, cercanía a la fecha de estancia, demanda histórica y nivel actual de reservas, sin asumir linealidad. La curva SHALL condicionarse por la temporada de la fecha de estancia (alta/media/baja) y por la ventana de feriado O evento privado que contenga la fecha — los eventos con propiedad asignada solo condicionan a esa propiedad — entrenarse con una ventana de recencia configurable y degradar con fallback explícito cuando un bucket no acumule evidencia suficiente. Los bins de lead SHALL soportar bordes explícitos configurables (`bin_edges`), densos a leads cortos donde la curva de reservas es más cóncava, con el último borde como catch-all.

#### Scenario: Gradiente fino a leads cortos
- **WHEN** la curva se ajusta con bordes densos a leads cortos (p. ej. [1, 3, 7, 14, 21, 30, …])
- **THEN** leads operativos adyacentes (p. ej. 1 vs 9) reciben pickups esperados distintos cuando los datos los tengan — sin comprimir el gradiente hacia la media del bin
- **THEN** los leads por encima del último borde caen en el bin catch-all (nunca NaN ni fuera de rango)

#### Scenario: Validación de bordes
- **WHEN** la configuración declara bordes no crecientes, con menos de 2 elementos o con el primer borde por debajo del lead mínimo
- **THEN** la configuración falla con error explícito antes de entrenar

#### Scenario: Trayectoria bookings → pickup → final
- **WHEN** se evalúa una fecha con distintos lead times
- **THEN** el comportamiento esperado sigue la trayectoria `Current Bookings → Expected Pickup → Final Room Nights` con pickup decreciente conforme el lead disminuye, salvo evidencia histórica en contrario

#### Scenario: Curva condicionada por temporada del stay
- **WHEN** dos estancias al mismo lead caen en temporadas distintas (p. ej. alta vs baja)
- **THEN** el pickup esperado de cada una refleja el ritmo de reserva de SU temporada
- **THEN** a leads largos el forecast converge al nivel estacional de la fecha de estancia, no al promedio anual

#### Scenario: Curva condicionada por ventana de feriado
- **WHEN** una estancia cae dentro de la ventana de un feriado o evento privado con suficiente evidencia histórica
- **THEN** su pickup esperado usa la curva de ese feriado/evento (p. ej. Semana Santa ≠ un marzo cualquiera; un congreso ≠ un martes cualquiera)
- **THEN** un bucket de feriado/evento sin evidencia suficiente degrada a la temporada de la fecha

#### Scenario: Evento con scoping por propiedad
- **WHEN** un evento declara una propiedad y otra propiedad tiene estancias en las mismas fechas
- **THEN** la otra propiedad NO recibe la curva de ese evento

#### Scenario: Ventana de recencia del fit
- **WHEN** la historia contiene regímenes antiguos fuera de la ventana de recencia configurada
- **THEN** las estancias fuera de la ventana no participan del entrenamiento de la curva

#### Scenario: Fallback por escasez de evidencia
- **WHEN** un bucket (propiedad, temporada, feriado o evento) acumula menos estancias que el mínimo configurado
- **THEN** el lookup degrada al siguiente nivel de condicionamiento disponible hasta la curva de la propiedad y, si tampoco existe, a la global
- **THEN** el fallback queda determinístico y documentado en la configuración de la corrida
