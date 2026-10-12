# Tasks — long-horizon-folds

## 1. Config y wiring

- [x] 1.1 `config.yaml` sección `benchmark` (`fold_horizon_days: 90`, `n_folds: 4`) + parseo en `src/config.py`. Verificar: test de config (defaults y valores custom).
- [x] 1.2 `src/forecasting/run_real.py`: `real_fold_metrics` con horizonte de fold 90 (tiling del año, helper `fold_cutoffs_for`) y wiring desde config; docstrings actualizados. Verificar: la corrida real imprime el benchmark con ventanas de 90d.
- [x] 1.3 `src/reporting/benchmark.py`: `run_benchmark(..., dm_horizon)` → `select_champion`; `src/forecasting/runner.py`: `_dm_horizon_from_panel(error_panel)` (mediana de fechas por fold; vacío → 30). Verificar: test unitario de la derivación.

## 2. Validación

- [x] 2.1 Tests (`tests/test_long_horizon_folds.py`): tiling de 4 ventanas de 90d sin solape entre folds ni con la ventana de reporte; cobertura ≥ 10 meses del año; 90 fechas distintas por fold en el panel; defaults de config; derivación del dm_horizon; threading en `run_benchmark`. Verificar: suite en verde.
- [x] 2.2 Corrida real de verificación con timing: TODO OK, benchmark con ventanas de 90d, DM con n_obs 360 (antes 120), selecciones por propiedad comparadas contra la corrida anterior (watch item: MARINA cambió catboost→lightgbm), bandas por lead. Verificar: artefactos inspeccionados y runtime documentado.
      Resultado (ventana 2026-07-09..10-06, mismos datos): WMAPE 0.1638 → **0.1577**, MAPE 26.91% → **25.10%**, Bias −0.038 → −0.031; MARINA 0.2800 → **0.1617**; bandas 30-59/60-89/90+ mejoran (0.182/0.184/0.245 → 0.175/0.171/0.229). Runtime 426s → 521s (+22%).
- [x] 2.3 Corrida futura TODO OK (412s): manifiesto con metrics_source injected_backtest y campeones coherentes — campeón global pickup_curve (empate DM p=0.482 con n=360), por propiedad FIESTA=seasonal_naive, MARINA=xgboost, resto pickup_curve; SJOSL en fallback por histórico corto. 15 alertas de capacidad (pico estacional) — manejadas (clamp) y reportadas.
- [x] 2.4 ~~Suite completa~~ en verde (265 tests) + ~~sync de specs~~ (`openspec/specs/evaluation/spec.md` actualizado). Falta: archive + commit (pendiente de aprobación del usuario).
