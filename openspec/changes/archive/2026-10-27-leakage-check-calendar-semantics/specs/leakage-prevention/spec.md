# leakage-prevention — delta

## MODIFIED Requirements

### Requirement: Pruebas automáticas de leakage
#### Scenario: Recalculo en la semántica del builder (calendario)
- **WHEN** se recalcula una feature rolling almacenada para contrastarla con la ventana solo-hacia-atrás
- **THEN** el recalculo usa la MISMA semántica del builder: la grilla calendario reindexada por propiedad (los huecos de la fuente son NaN que la ventana salta con min_periods), comparando solo en las fechas observadas
- **THEN** un recalculo por ENTRADAS en series con huecos genera falsos positivos (coincidencias numéricas con la ventana centrada por-entradas, caso MARINA 2025-05-09) y SHALL evitarse
