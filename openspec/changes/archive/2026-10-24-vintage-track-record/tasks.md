# Tasks — vintage-track-record

- [x] 1.1 `src/monitoring/accuracy_tracking.py`: `build_track_record` — deduplica por añada (published_run_id; fallback cutoff = window.start − 1d), cada añada con su graduación más reciente, ordenada por publicación. Verificar: `tests/test_track_record.py` (dedup, atribución por cutoff, orden, vacío, TS peor + cobertura).
- [x] 1.2 `src/reporting/visual_report.py`: tarjeta «Expediente del forecast publicado» (gráfica WMAPE por añada con n + tabla + guía) en Precisión; fila del panel con tendencia («Acumulando» < 2 añadas; mejora/estable/revisar con ±15% contra el promedio previo) y salvedad de n. Verificar: `test_panel_expediente_*` (3 tests).
- [x] 1.3 `src/forecasting/runner.py`: expediente desde el registro → reporte; resumen `track_record` en el manifiesto.
- [x] 1.4 `scripts/regen_report.py`: expediente as-of (registros con timestamp ≤ la corrida regenerada).
- [x] 1.5 Tests de los módulos afectados en verde (72 en track_record/reporting/accuracy/e2e/future_run).
- [x] 1.6 Suite completa + specs (monitoring + reporting) + este change archivado + commit.
