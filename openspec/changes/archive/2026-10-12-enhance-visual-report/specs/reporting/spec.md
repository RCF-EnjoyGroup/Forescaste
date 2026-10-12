## ADDED Requirements

### Requirement: Vista HTML por propiedad
El sistema SHALL incluir en `report.html` el comportamiento por propiedad: forecast de cada propiedad con su banda de incertidumbre, techo de capacidad y reales cuando existan; accuracy por propiedad con su modelo operante; y las revisiones del backtest T-90..T-1 desglosadas por propiedad y por banda de lead. Las secciones SHALL ser defensivas ante datos ausentes y jamás romper la corrida.

#### Scenario: Forecast por propiedad
- **WHEN** la corrida produce forecast para varias propiedades
- **THEN** el reporte muestra una gráfica por propiedad con su P50, banda P10/P90 y techo de capacidad
- **THEN** el título de cada panel declara el modelo que opera esa propiedad
- **THEN** los reales se superponen cuando la ventana tiene observados

#### Scenario: Accuracy por propiedad
- **WHEN** la evaluación calcula el desglose por propiedad
- **THEN** el reporte muestra el WMAPE por propiedad y una tabla con propiedad, modelo, WMAPE, MAPE, bias y n_obs

#### Scenario: Revisiones del backtest por propiedad
- **WHEN** el backtest T-90..T-1 produce revisiones
- **THEN** el reporte muestra el WMAPE de las revisiones por propiedad y la convergencia del error por banda de lead (una serie por propiedad)
- **THEN** el `backtesting_report.json` incluye el desglose `by_property_lead_band`

#### Scenario: Datos ausentes
- **WHEN** falta alguno de los desgloses o el backtest está vacío
- **THEN** las secciones correspondientes se omiten con un mensaje y la corrida no falla
