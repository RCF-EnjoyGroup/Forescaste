# Tasks — add-event-features

## 1. Fuente y calendario

- [x] 1.1 `src/features/events.py`: `load_events(path)` con validación dura (fechas legibles, duplicados nombre+fecha+propiedad, normalización de property opcional) → None si el archivo no existe. Verificar: tests de carga válida/ausente/inválida.
- [x] 1.2 `src/features/holidays.py`: `HolidayCalendar` gana `holiday_dates` y `scoped`; `window_key(date, property=None)`; `calendar_with_events(base, events)` combinando feriados + eventos (fechas y ventanas). Verificar: tests de combinación, scoping y de que `is_holiday` no marca eventos.
- [x] 1.3 `config.yaml` `data.events_path` + parseo en `src/config.py`. Verificar: test de config.

## 2. Modelo y wiring

- [x] 2.1 `src/models/baselines.py`: fit y `_curve_for` pasan la propiedad a `window_key` (scoping). Verificar: test de que un evento scoped de otra propiedad no contamina la curva.
- [x] 2.2 `runner.py` + `run_real.py`: carga del archivo de eventos (si existe), calendario combinado al forecast/benchmark/backtest, manifiesto con `events: {source, n_events}`. Verificar: test e2e con archivo sintético en tmp_path (manifiesto lo declara) y sin archivo (declara ausencia).
- [x] 2.3 `README_SISTEMAS.md`: contrato de exportación de eventos (formato, columnas, ejemplo, checklist). Verificar: revisión del texto.

## 3. Cierre

- [x] 3.1 Suite completa en verde + corrida de verificación SIN events.csv produce métricas idénticas a las del cambio anterior (inertes-sin-datos). Verificar: comparación global/bandas.
- [x] 3.2 Sync de specs + archive + commit.
