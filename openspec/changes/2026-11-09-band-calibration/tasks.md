# Tasks — band-calibration

- [ ] 1.0 **Investigación previa (bloqueante)**: medir por qué la cobertura observada (66-96% por propiedad) se desvía del nominal 80% cuando la banda se construye como cuantiles del residuo operativo. Hipótesis: (a) clipping por capacidad/piso de reservas recorta el p10/p90 publicado; (b) la muestra de folds (año previo) no representa la ventana publicada (régimen); (c) otra. Medir por propiedad: |error| real vs semiancho publicado.
- [ ] 1.1 `src/forecasting/core.py`: helper `calibrate_band_by_property` según causa medida en 1.0 (factor multiplicativo al semiancho con shrink hacia 1.0 por pocas obs); knob `interval_calibration` en config (default OFF).
- [ ] 1.2 `src/forecasting/runner.py`: con knob ON, aplicar el factor al semiancho P10-P90 publicado y volcar `interval_calibration.cells` (factor + cobertura observada por propiedad) al manifiesto.
- [ ] 1.3 Tests: (a) knob OFF → banda idéntica a la actual; (b) knob ON sobre backtest etiquetado → cobertura por propiedad se acerca al nominal; (c) shrink: propiedad con 3 obs no recibe factor extremo.
- [ ] 1.4 A/B de replay (3 brazos OFF/débil/fuerte) y decisión de default; recién entonces proponer default ON.
- [ ] 1.5 Delta spec forecasting, change archivado, commit.
