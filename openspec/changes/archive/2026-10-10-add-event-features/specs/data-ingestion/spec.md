## ADDED Requirements

### Requirement: Fuente opcional de eventos
El sistema SHALL soportar una fuente opcional de eventos privados (`data.events_path`, columnas `event_name`, `event_date` y `property` opcional); cuando el archivo exista SHALL validarse (fechas legibles, sin duplicados por nombre+fecha+propiedad) y alimentar el calendario compartido, y cuando no exista el feature SHALL omitirse con la ausencia documentada en la corrida.

#### Scenario: Archivo de eventos presente
- **WHEN** `events.csv` existe y es válido
- **THEN** las fechas de eventos entran al calendario de la corrida (feature de distancia de ML y ventanas de la curva)
- **THEN** el manifiesto declara la fuente activa con su número de eventos

#### Scenario: Archivo de eventos ausente
- **WHEN** `events.csv` no existe
- **THEN** la corrida es idéntica a una sin eventos y el manifiesto declara la fuente ausente

#### Scenario: Archivo inválido
- **WHEN** el archivo contiene fechas ilegibles o duplicados por nombre+fecha+propiedad
- **THEN** la carga falla con un mensaje que identifica el problema ANTES de entrenar
