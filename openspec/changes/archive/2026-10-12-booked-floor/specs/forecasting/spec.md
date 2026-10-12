## MODIFIED Requirements

### Requirement: Restricción de capacidad física
El sistema SHALL garantizar que el forecast final respete el piso de reservas en libros y el techo de capacidad: `final_forecast = min(max(raw_forecast, current_booked_rn), capacity)` cuando existe on-the-books, y `final_forecast = min(raw_forecast, capacity)` cuando no. Las restricciones son operacionales del resultado, no variables predictivas.

#### Scenario: Aplicación del techo
- **WHEN** el forecast estadístico/ML supera la capacidad para una fecha
- **THEN** `final_forecast = min(raw_forecast, capacity)`
- **THEN** se conservan por separado `raw_forecast`, `capacity` y `final_forecast`

#### Scenario: Piso de reservas en libros
- **WHEN** el forecast del campeón queda por debajo de las reservas ya confirmadas as-of el corte (`current_booked_rn` conocido)
- **THEN** `final_forecast = max(raw_forecast, current_booked_rn)` sujeto al techo de capacidad
- **THEN** el p50 coincide con el punto publicado (el piso aplica al punto, no a la banda)

#### Scenario: Sin on-the-books
- **WHEN** la estancia no tiene snapshot de reservas al corte
- **THEN** no se aplica piso y el forecast queda solo sujeto al techo

#### Scenario: Banda no pisada
- **WHEN** el piso elevó el punto publicado
- **THEN** la banda P10/P90 puede quedar por debajo del booked (las cancelaciones son resultado posible) y solo se restringe a `[0, capacity]`

#### Scenario: Booked por encima de la capacidad
- **WHEN** las reservas confirmadas superan la capacidad disponible (inconsistencia de datos)
- **THEN** el techo de capacidad prevalece sobre el piso y el sistema emite una alerta `booked_over_capacity`

#### Scenario: Alerta de violación
- **WHEN** `raw_forecast > capacity`
- **THEN** el sistema emite una alerta de constraint violation antes de aplicar el cap
