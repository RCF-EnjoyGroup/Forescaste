# Tasks — seasonal-calibration-cells

- [x] 1.1 `src/evaluation/breakdowns.py`: `season_of_stay()` como fuente única del mapeo (refactor de `metrics_by_seasonality`, comportamiento idéntico).
- [x] 1.2 `src/forecasting/core.py`:
      (a) `lead_bias_factors(..., seasonal_cells=False)` estima por (prop, banda, temporada) con los mismos gates y fallback honesto estacional → banda; reporte por celda con fuente aplicada (`season`/`band`/`none`).
      (b) `apply_bias_factors` con llaves (prop, banda, temporada), mapeo por fila del horizonte y fallback de banda SOLO en combinaciones sin llave estacional.
      (c) `operative_residual_bands(..., seasonal_cells=False)`: llaves 3-tupla y residuo con el factor fuente por fila.
      (d) `build_intervals`: modo 3-tupla con ladder celda-estacional → (prop,banda) → prop → banda → global, todo documentado en `note`.
- [x] 1.3 `src/forecasting/runner.py` + `config.yaml`: knob `bias_correction.seasonal_cells` cableado una sola vez, manifiesto expone knob y detalle (season_cells incluidos por serialización existente de bias_report).
- [x] 1.4 Tests: `tests/test_seasonal_calibration.py` (14 tests: knob OFF bit a bit, gates estacionales, fallback declarado, contratos). Suite completa 436 passed + 1 skipped en verde.
- [x] 1.5 A/B con data real: `scripts/ab_seasonal_cells.py` — baseline `real-20261030-1147` (off) vs `real-20261030-1253` (on). **RESULTADO NEGATIVO**: FIESTA 42.2%→23.3%, CORIN −12.2pp, WMAPE global 0.1767→0.1817. El mecanismo aplicó correctamente (43 celdas con fuente en el manifiesto); la hipótesis falló porque el sesgo es drift de nivel YoY, no bias estacional recurrente. **Decisión: default `seasonal_cells: false`** con evidencia documentada en config. Dos defectos atrapados en el proceso y corregidos: anidamiento del knob en el YAML (experimento nulo #1) y llaves-tupla en la serialización del manifiesto (crash #2).
- [x] 1.6 Spec forecasting (delta con decisión), change archivado, commit.
