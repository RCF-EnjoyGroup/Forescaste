# Tasks — bias-correction-mode

- [x] 1.1 `src/forecasting/core.py`: `fold_recency_decay` en `lead_bias_factors`/`_bias_cell_estimate` — pesos `decay**(max_fold−fold)` en `rel` y consistencia ponderada; `decay=1.0` bit a bit.
- [x] 1.2 `src/forecasting/runner.py`: knob `bias_correction.enabled` — capa omitible con reporte honesto al manifiesto; bandas con factores vacíos.
- [x] 1.3 `config.yaml`: knobs con comentario (defaults = comportamiento actual).
- [x] 1.4 Tests `tests/test_bias_mode.py` (6 pruebas: decay=1 invariante, régimen reciente domina, consistencia ponderada, min_obs crudas, apply sin factores, árbol del config). Suite completa en verde.
- [x] 1.5 A/B 3 brazos (misma data): **defaults SIN cambio** — apagar la capa empeora SJOSL (+4.7pp sesgo, +1.8pp wmape); la mejora de MARINA sin capa quedó confundida por la cadena de histéresis del registro (`futuro-20261104-1525` giró el incumbente a sarimax); la recencia 0.7 no daña pero no cumple la mejora estricta. Veredicto y evidencia documentados en config y proposal.
- [x] 1.6 Delta spec forecasting (escenarios kill-switch y recencia con decisión), change archivado, README, commit.
