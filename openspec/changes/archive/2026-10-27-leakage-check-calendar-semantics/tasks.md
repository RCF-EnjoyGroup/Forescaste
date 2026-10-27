# Tasks — leakage-check-calendar-semantics

- [x] 1.1 `src/validation/leakage.py`: `check_rolling_with_future` recalcula backward/forward sobre la grilla calendario reindexada (deduplicación defensiva por fecha, comparación solo en fechas observadas) — la MISMA semántica del builder.
- [x] 1.2 Tests de regresión: `test_rolling_calendario_correcto_con_hueco_no_es_fuga` (el caso EXACTO de MARINA 05-09: hueco en 05-06, stored 7.000 calendario-correcto → sin hallazgo) y `test_rolling_fuga_real_con_hueco_sigue_detectada` (stored = centrada en grilla → hallazgo). El test existente de fuga real sin huecos sigue pasando.
- [x] 1.3 Suite leakage + features + e2e + modeling + remediation en verde (92 tests).
- [x] 1.4 Spec leakage-prevention actualizado (scenario de recalculo en semántica del builder) + este change archivado + commit. Corrida real queda desbloqueada para el usuario.
