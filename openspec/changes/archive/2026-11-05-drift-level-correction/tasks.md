# Tasks — drift-level-correction

- [x] 1.1 `src/forecasting/core.py`: `level_correction_factors(detail, *, trigger_ts, window_days, min_obs, shrinkage, max_rel)` — ventana reciente, (n, rel, ts) por propiedad, peso suave `min(|ts|/trigger,1)`, shrinkage, clip, razón por celda; `apply_level_factors(frame, factors)` con columna `level_correction_factor`.
- [x] 1.2 `src/forecasting/runner.py`: cableado en `run_end_to_end` tras `build_forecasts` — knob `monitoring.level_correction`, SOLO `run_mode == "futuro"` con publicación previa operativa; reporte al manifiesto; razón documentada cuando no aplica.
- [x] 1.3 `config.yaml`: knob con comentario de diseño (default lo fija el A/B).
- [x] 1.4 Tests `tests/test_level_correction.py` (10 pruebas: peso suave, gates n/clip/inmaterial, aplicador, composición con banda). Suite completa 451 tests en verde.
- [x] 1.5 A/B replay walk-forward `scripts/ab_level_correction.py`: **RESULTADO NEGATIVO** — la señal solo observa leads cortos ya calibrados (FIESTA +0.2%); al disparar sobre-corrigió (cobertura FIESTA 90.8%→81.7%). **Decisión: default `enabled: false`** con evidencia documentada en config; re-apertura con señal de publicación de 45-60 días cuando el registro la permita validar.
- [x] 1.6 Deltas de spec (monitoring con decisión), change archivado, README, commit.
