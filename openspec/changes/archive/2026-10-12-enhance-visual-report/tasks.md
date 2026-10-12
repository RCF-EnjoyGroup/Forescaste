# Tasks — enhance-visual-report

- [x] 1.1 `src/forecasting/runner.py`: `_sampled_backtest` emite `by_property_lead_band` (property × banda → n_obs, WMAPE, MAPE). Verificar: test con fixture multi-propiedad.
- [x] 1.2 `src/reporting/visual_report.py`: secciones "Forecast por propiedad" (small multiples con banda/capacidad/reales + modelo en título), "Accuracy por propiedad" (barras + tabla), "Revisiones T-90..T-1" (barras por propiedad + convergencia por banda + tabla por banda); KPIs Campeón y Propiedades evaluadas n/N; kwargs opcionales y defensivos. Verificar: tests con y sin datos opcionales.
- [x] 1.3 Call site del runner: pasar `property_breakdown`, `model_by_property`, `backtest_report`, `champion`. Verificar: test e2e genera HTML con las secciones nuevas.
- [x] 1.4 Suite completa en verde + corrida real con `report.html` inspeccionado (secciones presentes) + README actualizado + sync/archive + commit.
