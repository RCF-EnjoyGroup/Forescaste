# Tasks — improve-long-horizon-forecast

## 1. Exclusión de propiedades cerradas

- [x] 1.1 `config.yaml` + `src/config.py`: clave `data.excluded_properties` (lista, default vacía) parseada a `EngineConfig`. Verificar: test de config con la clave presente/ausente.
- [x] 1.2 `src/forecasting/run_real.py` (`load_real_sources`) y `src/forecasting/runner.py` (`run_end_to_end`): filtro de las tres fuentes tras aplicar alias, antes de calidad; el manifiesto registra `excluded_properties` y el coverage las separa de las excluidas por ventana. Verificar: test e2e donde una propiedad excluida no aparece en forecast/benchmark/coverage y el manifiesto la declara.

## 2. Curva de pickup estacional con recencia

- [x] 2.1 `src/models/baselines.py`: `PickupCurveModel` con `season_conditional`, `max_history_days`, `min_stays_per_bucket`; curvas por (propiedad, temporada) con fallback propiedad → global; lookup por temporada del `stay_date` del request. Verificar: tests unitarios de condicionamiento (temporadas distintas al mismo lead difieren), recencia (stays fuera de ventana no influyen) y fallback (bucket delgado degrada).
- [x] 2.2 `src/forecasting/core.py`: `curve_params` en `fit_candidate`/`build_forecasts`; `src/forecasting/run_real.py` y `src/forecasting/runner.py` pasan los parámetros del config al benchmark, forecast y backtest. Verificar: test de que los parámetros del config llegan al modelo operante.
- [x] 2.3 Defaults en `config.yaml` (`season_conditional`, `max_history_days`, `min_stays_per_bucket`) fijados tras el A/B (tarea 3.1). Verificar: config documentada y corrida real usa los defaults.

## 3. Validación y corridas

- [x] 3.1 Experimento A/B (script desechable): curva estática vs estacional/reciente al corte 2026-07-08 sobre las 6 propiedades activas, comparación MAPE/WMAPE por banda de lead de la ventana Jul–Oct + backtest por lead. Verificar: decisión documentada con números.
- [x] 3.2 Suite completa en verde (`python -m pytest -q`). Verificar: 0 fallos.
- [x] 3.3 Corridas real y futuro con LIRAK/LAPAS excluidas y curva final: TODO OK, bandas 30–89 mejoradas vs corrida anterior, LIRAK/LAPAS ausentes de todos los entregables. Verificar: inspección de artefactos y comparación por banda.
