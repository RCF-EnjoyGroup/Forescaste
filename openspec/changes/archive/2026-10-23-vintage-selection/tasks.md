# Tasks — vintage-selection

- [x] 1.1 `src/monitoring/accuracy_tracking.py`: `select_gradable_publication` — itera registros `futuro` reciente→antiguo (máx. 5), devuelve la primera publicación cuyo `score_vintage` no es None, con el record para trazabilidad. Verificar: `test_select_gradable_publication_salta_la_no_graduable` (caso real: salta la futura sin reales, gradúa la anterior), `test_select_gradable_publication_sin_graduables`.
- [x] 1.2 `src/forecasting/runner.py`: el bloque vintage usa la selección graduable (vintage + detail CSV con el `published_run_id` de la añada graduada); `forecast_vs_previous` sigue contra la última publicación (linaje operativo, sin cambio). Verificar: e2e + future_run en verde.
- [x] 1.3 Panel: la fila vintage cita la añada graduada (`published_run_id`) en su criterio.
- [x] 1.4 Tests de los módulos afectados en verde (65 en accuracy_tracking/reporting/e2e/future_run).
- [x] 1.5 Suite completa + specs + este change archivado + commit.
