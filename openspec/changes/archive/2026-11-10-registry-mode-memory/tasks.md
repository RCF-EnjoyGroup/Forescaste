# Tasks — registry-mode-memory

- [x] 1.1 `src/forecasting/runner.py`: `previous_run = latest_record(registry, mode=str(run_mode))` — los tres consumidores con memoria (histéresis, drift, retraining) ven su propia línea; sin registro del modo → memoria vacía declarada.
- [x] 1.2 Tests `tests/test_registry_memory.py`: registro mixto → la real toma la última real aunque la futura sea más reciente; sin registro del modo → None; línea futura intacta; ad hoc aislado.
- [x] 1.3 Replay `scripts/ab_registry_memory.py` sobre el registro real: previous_run old-vs-new, con el caso 1105-1442 documentado.
- [x] 1.4 Delta spec monitoring (escenario de memoria por modo con la validacion del replay), change archivado, commit y push.

