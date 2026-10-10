## ADDED Requirements

### Requirement: Propiedades cerradas excluidas
El sistema SHALL soportar la exclusión configurable de propiedades cerradas (`data.excluded_properties`); las propiedades excluidas no participan de ingesta, calidad, features, benchmark, forecast, backtest ni KPIs, y la exclusión SHALL quedar registrada en el manifiesto.

#### Scenario: Exclusión aplicada en toda la corrida
- **WHEN** una propiedad está en `excluded_properties`
- **THEN** no aparece en la tabla de forecast, el benchmark, el backtest ni la cobertura de evaluación
- **THEN** el manifiesto declara la lista de propiedades excluidas

#### Scenario: Exclusión tras alias
- **WHEN** una propiedad cerrada aparece bajo un alias distinto en alguna fuente
- **THEN** el alias se resuelve primero y la exclusión se aplica al nombre canónico
- **THEN** ninguna fila de la propiedad excluida cruza al pipeline
