## Why

El blend por lead (cambio `gbm-blend-long-leads`, commit `acd5313`) reemplaza los leads > 30 del campeón ML por pickup_curve cuando el auto-diagnóstico detecta colapso del rollout (q25 ratio < 0.85). El trigger es correcto pero **accuracy-blind**: nunca verifica que el reemplazo mejore la precisión.

La validación de FIESTA (2026-10-14) contra los reales de la ventana y contra los folds del propio motor lo demuestra:

- El blend está activo solo para FIESTA (`per_property_blend` del manifiesto) y reemplaza lightgbm largo (wmape_long **11.3%**) por pickup_curve largo (**26.3%** — el PEOR modelo largo del tablero, 2.3× peor que lo que "rescata").
- Efecto visible (lo que motivó la validación): a leads > 30, los 14 viernes/sábados tienen raw +60 RN sobre la capacidad (máx +123) → recortados al techo 408 ("el rango se pierde" en la cima), mientras lunes-jueves se desploman −39 a −87 RN contra reales.
- La métrica T0 del run no lo delató porque el techo enmascaró el sobre-pronóstico de findes (los reales también ≈ 408); la aceptación original del blend no incluyó el desglose por lead ni los folds por modelo.
- Los folds (4 trimestrales, al corte del run) dicen que los mejores largos de FIESTA son sarimax (10.2%) y catboost (10.8%) — no pickup_curve.

## What Changes

- **Selección del reemplazo largo por evidencia**: cuando el colapso del rollout se detecta, el reemplazo de leads > 30 pasa de ser pickup_curve por defecto a ser **el candidato con mejor `wmape_long` en los folds de esa propiedad**, y solo si es materialmente mejor que el propio campeón (`wmape_long_alt < wmape_long_campeón × BLEND_ACCURACY_MARGIN`, margen 0.95). Si ningún candidato califica, el campeón conserva su rollout completo (el colapso era distribucional, no de precisión).
- **Sin evidencia, no hay cirugía**: si la corrida no tiene fold metrics con `wmape_long` (corridas de muestra), el colapso se registra pero no se reemplaza.
- **Cableado**: `build_forecasts` recibe `fold_metrics` (el runner ya los computa); el manifiesto enriquece `per_property_blend` con `applied` y `long_selection` (la evidencia numérica de la decisión).
- La detección del colapso (q25 ratio < 0.85, split 30) queda igual.

## Capabilities

### Modified Capabilities
- `forecasting`: MODIFIED — Champion/challenger: el blend por lead es accuracy-aware (el reemplazo largo se elige por evidencia de folds, no por defecto).

## Impact

- **Código**: `src/forecasting/core.py` (`BLEND_ACCURACY_MARGIN`, `_select_long_replacement`, `build_forecasts` con `fold_metrics`), `src/forecasting/runner.py` (pasa los fold metrics), `tests/test_gbm_blend.py` (extendido).
- **Cableado**: `run_real.py` ya computa los folds y los pasa al runner — sin cambios.
- **Comportamiento observable esperado (FIESTA con los datos actuales)**: el blend pasa de curve (26.3% largo) a sarimax (10.2%) en leads > 30; desaparece el recorte al techo de los findes (sarimax nunca excede capacidad) y el "rango perdido"; el T0 por banda queda ~15.8%/13.1% (vs 16.0%/13.2% con curve) y la robustez a largo lead mejora 2.3× según folds.
- **Riesgo**: bajo — sin colapso no hay cambio (5 de 6 propiedades no se tocan); el mecanismo es general (cualquier propiedad con campeón ML colapsado usa la misma regla de evidencia); las corridas sin folds quedan sin blend (comportamiento conservador documentado).
