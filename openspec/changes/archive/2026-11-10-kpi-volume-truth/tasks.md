# Tasks — kpi-volume-truth

- [x] 1.1 `src/forecasting/runner.py`: calcular métricas excluyendo cold-start (`global_excl_cold_start` + lista de excluidas con su razón) junto al global; KPIs etiquetados nuevos.
- [x] 1.2 `src/reporting/visual_report.py`: nota de auditoría con la línea «excluyendo cold-start» cuando aplique.
- [x] 1.3 Tests: con cold-start el bloque aparece y difiere; sin cold-start no aparece. (tests/test_kpi_volume_truth.py, 3 tests)
- [ ] 1.4 Delta spec reporting, change archivado, commit.
