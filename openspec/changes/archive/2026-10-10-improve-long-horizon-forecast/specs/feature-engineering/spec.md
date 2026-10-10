## MODIFIED Requirements

### Requirement: Curvas de pickup no lineales
El sistema SHALL modelar el pickup como una curva no lineal por lead time, propiedad, día de semana, mes, temporada, cercanía a la fecha de estancia, demanda histórica y nivel actual de reservas, sin asumir linealidad. La curva SHALL condicionarse por la temporada de la fecha de estancia (alta/media/baja), entrenarse con una ventana de recencia configurable y degradar con fallback explícito cuando un bucket estacional no acumule evidencia suficiente.

#### Scenario: Trayectoria bookings → pickup → final
- **WHEN** se evalúa una fecha con distintos lead times
- **THEN** el comportamiento esperado sigue la trayectoria `Current Bookings → Expected Pickup → Final Room Nights` con pickup decreciente conforme el lead disminuye, salvo evidencia histórica en contrario

#### Scenario: Curva condicionada por temporada del stay
- **WHEN** dos estancias al mismo lead caen en temporadas distintas (p. ej. alta vs baja)
- **THEN** el pickup esperado de cada una refleja el ritmo de reserva de SU temporada
- **THEN** a leads largos el forecast converge al nivel estacional de la fecha de estancia, no al promedio anual

#### Scenario: Ventana de recencia del fit
- **WHEN** la historia contiene regímenes antiguos fuera de la ventana de recencia configurada
- **THEN** las estancias fuera de la ventana no participan del entrenamiento de la curva

#### Scenario: Fallback por escasez de evidencia
- **WHEN** un bucket (propiedad, temporada) acumula menos estancias que el mínimo configurado
- **THEN** el lookup degrada a la curva de la propiedad y, si tampoco existe, a la global
- **THEN** el fallback queda determinístico y documentado en la configuración de la corrida
