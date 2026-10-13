# Tasks — gbm-blend-long-leads

- [x] 1.1 `src/forecasting/core.py`: `BLEND_LEAD_SPLIT = 30`; `build_forecasts` ajusta pickup_curve adicional cuando el campeón es GBM y combina por lead (≤30 GBM, >30 curva). Verificar: test de que leads cortos usan GBM, largos usan curva.
- [x] 1.2 `src/forecasting/runner.py`: manifiesto `per_property_blend` con propiedades que operan blend y sus modelos. Verificar: test e2e con campeón GBM.
- [x] 1.3 Suite completa + corrida real: FIESTA sin caídas de septiembre, sin regresión en otras propiedades. Verificar: comparación por propiedad.
- [x] 1.4 Sync specs + archive + commit.
