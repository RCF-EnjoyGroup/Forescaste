## Why

`report.html` solo muestra el portafolio agregado: la operación es por propiedad (cada hotel se pronostica con su campeón) y ni el forecast por propiedad, ni el accuracy por propiedad, ni las revisiones T-90..T-1 (backtest muestreado) aparecen en la vista de revisión. El negocio revisa el reporte por hotel, no por portafolio.

## What Changes

- **Forecast por propiedad**: small multiples (una mini-gráfica por propiedad) con P50, banda P10/P90, techo de capacidad y reales cuando existen; el título de cada panel declara el modelo que opera esa propiedad.
- **Accuracy por propiedad**: barras de WMAPE + tabla (propiedad, modelo, WMAPE, MAPE, bias, n_obs) desde el desglose que ya calcula la corrida.
- **Revisiones T-90..T-1 por propiedad**: WMAPE del backtest muestreado por propiedad + gráfica de convergencia (WMAPE por banda de lead, una línea por propiedad — cómo converge cada hotel conforme se acerca la estancia) + tabla por banda. Requiere que `_sampled_backtest` emita el desglose `by_property_lead_band` (property × banda).
- **KPIs ampliados**: tarjeta "Campeón" y "Propiedades evaluadas (n/N)".
- Todas las secciones siguen siendo defensivas: sin datos → se omiten sin romper la corrida.

## Capabilities

### New Capabilities

<!-- Ninguna. -->

### Modified Capabilities

- `reporting`: ADDED — Vista HTML por propiedad: `report.html` SHALL incluir el comportamiento por propiedad (forecast con banda y modelo, accuracy por propiedad, revisiones del backtest por propiedad y por banda de lead), defensivas ante datos ausentes.

## Impact

- **Código**: `src/reporting/visual_report.py` (secciones + KPIs), `src/forecasting/runner.py` (`by_property_lead_band` en el reporte del backtest + call site con los nuevos datos), `tests/test_reporting.py`.
- **Comportamiento observable**: `report.html` pasa de 6 secciones de portafolio a incluir 4 vistas por propiedad; el `backtesting_report.json` gana la clave `by_property_lead_band`.
- **Riesgo**: tamaño del HTML (más imágenes embebidas) — se mantiene autocontenido; sin datos opcionales el reporte queda como hoy.
