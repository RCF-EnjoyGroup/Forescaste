# Tasks — lead-weighted-composite

- [x] 1.1 `src/forecasting/run_real.py`: `real_fold_metrics` emite `wmape_short` (leads 1-30) y `wmape_long` (leads 31-89) por fila de (model, fold, property) — tanto por propiedad como en la fila de portafolio. Verificar: test con fixture que produce ambas columnas.
- [x] 1.2 `src/evaluation/selection.py`: `score_candidates` usa `LEAD_ALPHA·wmape_short + (1-LEAD_ALPHA)·wmape_long` como WMAPE del composite cuando ambas columnas existen; sin ellas cae al WMAPE plano. Constantes `LEAD_ALPHA=0.65`, `LEAD_SPLIT=30` documentadas. Verificar: test unitario del composite ponderado vs plano.
- [x] 1.3 Suite completa + corrida real: TODO OK; FIESTA cambia de campeón; NINGUNA propiedad regresa (verificar todas contra la corrida anterior). Verificar: comparación propiedad por propiedad.
- [x] 1.4 Sync specs + archive + commit.
