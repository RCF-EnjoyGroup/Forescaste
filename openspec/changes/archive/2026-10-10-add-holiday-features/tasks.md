# Tasks — add-holiday-features

## 1. Calendario de feriados

- [x] 1.1 `requirements.txt` +`holidays`; `src/features/holidays.py`: `build_holiday_calendar(years, countries, window_days)` con agrupación semántica (semana_santa, fin_de_ano, thanksgiving…), `HolidayCalendar` con `dates` y `windows` (empate → feriado más cercano), cache. Verificar: tests de fechas movibles (Semana Santa 2026 = 2–5 abr; Thanksgiving 2026 = 26 nov; Memorial 2026 = 25 may), ventanas y empates.
- [x] 1.2 `config.yaml` sección `holidays` (countries/window_days) + parseo en `src/config.py`. Verificar: test de config con países y deshabilitado (`[]` → calendario None).

## 2. Modelo y features

- [x] 2.1 `src/models/baselines.py`: `holiday_conditional` + `holiday_calendar` en `PickupCurveModel`; bucket (propiedad, feriado) con `min_stays_per_bucket`, cadena feriado → temporada → propiedad → global, ambos sujetos a `season_min_lead`. Verificar: tests de selección de curva por ventana de feriado, fallback por bucket delgado y umbral de lead.
- [x] 2.2 `src/forecasting/core.py`: `holiday_calendar` en `fit_candidate`/`build_forecasts`/`predict_candidate`; `days_to_nearest_holiday` en `_ml_row` y `ML_FEATURES`. Verificar: test de que la matriz ML contiene la feature y es correcta (0 en el feriado, k a k días).
- [x] 2.3 `src/forecasting/runner.py` + `run_real.py`: construcción del calendario desde config (rango de años de demanda + horizonte), paso a forecast/benchmark/backtest, `is_holiday` en el frame de features vía `add_calendar_features(holidays=...)`, manifiesto con config de feriados. Verificar: test e2e con feriados activos (`is_holiday` en reporte de features, manifiesto lo declara).

## 3. Validación y cierre

- [x] 3.1 A/B: estándar (ventana de verificación, F vs F+feriado) + dirigido (ventanas grandes con reales: Semana Santa 25/26, fin de año 25, Thanksgiving 25, a leads 45–75). Verificar: decisión del default documentada con números.
- [x] 3.2 Suite completa en verde + corridas real/futuro TODO OK con feriados operando. Verificar: 0 fallos; artefactos inspeccionados.
- [x] 3.3 Sync de specs + archive del cambio + commit.
