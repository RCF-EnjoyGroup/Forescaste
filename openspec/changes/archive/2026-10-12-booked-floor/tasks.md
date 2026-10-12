# Tasks — booked-floor-and-benchmark-viability

- [x] 1.1 `src/forecasting/constraints.py`: `apply_capacity_constraint(booked_col=...)` con `final = min(max(raw, booked), capacity)`, alertas de piso y de `booked_over_capacity`; booked NaN → sin piso. Verificar: tests unitarios de piso/conflicto/sin-booked.
- [x] 1.2 `src/reporting/forecast_table.py`: `validate_forecast_table` verifica la fórmula floor+techo (booked presente) y min(raw,cap) (booked ausente). Verificar: test de validación con caso pisado.
- [ ] 1.3 `src/forecasting/core.py`: `MIN_CURVE_STAYS = 60` — `_fit_pickup` degrada a nivel medio con <60 estancias. Verificar: test con pool de 30 stays → cold start.
- [ ] 1.4 `src/forecasting/run_real.py`: skip de candidatos cold-start en `real_fold_metrics` + fila de portafolio solo con viabilidad completa. Verificar: test con propiedad de historia corta (curva sin fila en fold temprano, con fila en folds tardíos).
- [x] 1.5 `src/forecasting/runner.py`: alertas/manifiesto (`n_booked_floor_alerts`, `n_booked_over_capacity`), doc técnica, README. Verificar: e2e con manifiesto declarando ambos.
- [x] 1.6 Suite completa + corrida real: TODO OK, FIESTA sin forecast bajo booked, selección de FIESTA revisada, métricas comparadas contra la corrida anterior. Verificar: artefactos + A/B.
- [x] 1.7 Sync specs + archive + commit.
