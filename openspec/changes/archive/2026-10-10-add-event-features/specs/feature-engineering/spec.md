## MODIFIED Requirements

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

### Requirement: Curvas de pickup no lineales
El sistema SHALL modelar el pickup como una curva no lineal por lead time, propiedad, día de la semana, mes, temporada, cercanía a la fecha de estancia, demanda histórica y nivel actual de reservas, sin asumir linealidad. La curva SHALL condicionarse por la temporada de la fecha de estancia (alta/media/baja) y por la ventana de feriado O evento privado que contenga la fecha — los eventos con propiedad asignada solo condicionan a esa propiedad — entrenarse con una ventana de recencia configurable y degradar con fallback explícito cuando un bucket no acumule evidencia suficiente.

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
