# Tasks — monthly-report-operative-models

- [x] 1.1 `scripts/monthly_vintage_report.py`: `_modelos_por_corte(records, corte)` con anclaje desde el registro (modo futuro, cutoff ≤ corte; pre-motor → registro más antiguo; sin registro → fallback declarado); `MODELOS` pasa a `MODELOS_FALLBACK`.
- [x] 1.2 Sello de versión en la hoja de lectura (generado, data hasta, regla, modelos por propiedad) + línea de "reales al" calculada de la data.
- [x] 1.3 Tests `tests/test_monthly_vintage_models.py`: anclaje por corte, pre-motor, sin registro, sello.
- [x] 1.4 Regenerar el entregable con la data vigente (octubre cerrado); confirmar pronósticos bit a bit en cortes existentes.
- [x] 1.5 Delta spec reporting (escenario de seleccion anclada + sello), change archivado, commit y push.

