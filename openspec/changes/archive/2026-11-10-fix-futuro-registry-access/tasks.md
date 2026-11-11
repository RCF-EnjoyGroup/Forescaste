# Tasks — fix-futuro-registry-access

- [x] 1.1 `src/forecasting/run_real.py`: helper `_incumbente_previo(cfg, mode="futuro")` (acceso dict a la ruta + filtro de modo) y reemplazo del bloque que crasheaba.
- [x] 1.2 Tests `tests/test_futuro_registry_access.py`: acceso dict, sin ruta, registro ausente, filtro modo=futuro; el test falla con AttributeError contra el código viejo.
- [x] 1.3 Re-corrida `--futuro` de verificación (llega a publicar sin crashear).
- [x] 1.4 Delta spec monitoring (escenario de memoria por línea cubre el modo futuro), change archivado, commit y push.
