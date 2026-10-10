## Why

La evaluación de la corrida de verificación muestra que el error a largo plazo domina el MAPE global del horizonte de 90 días (bandas 30–59: 27.8%, 60–89: 42.8%, 90+: 57.9%, contra 6.5% en 1–2 días). La causa estructural: el campeón `pickup_curve` entrena una curva **estática por propiedad** — media de pickup restante por bin de lead sobre TODA la historia y todo el año — de modo que a leads largos, donde el on-the-books cubre poco, el forecast converge al promedio anual en vez de al nivel estacional de la fecha de estancia. Además, dos de las ocho propiedades (LIRAK, LAPAS) están **cerradas**: sus datos están rancios o llenos de huecos y degradan benchmark, backtest y KPIs del portafolio.

## What Changes

- **Exclusión de propiedades cerradas (configurable)**: nueva clave `data.excluded_properties` en `config.yaml`; LIRAK y LAPAS quedan excluidas de ingesta, calidad, features, benchmark, forecast, backtest y KPIs. La exclusión se registra en el manifiesto y en el accuracy report (no simplemente desaparecen).
- **Curva de pickup condicionada por temporada**: `PickupCurveModel` aprende curvas por (propiedad, temporada de la fecha de estancia — high/shoulder/low) con fallback a la curva de propiedad y a la global cuando un bucket no acumula `min_stays_per_bucket` estancias. A leads largos el pickup esperado refleja el ritmo de reserva de la temporada del stay, no el promedio anual.
- **Ventana de recencia del fit**: las curvas se entrenan con las estancias de los últimos `max_history_days` días (default 1095), amortiguando regímenes viejos (p. ej. comportamiento pre/post-pandemia) sin sacrificar el volumen necesario para los buckets estacionales.
- **Ambas mejoras son parámetros del modelo operante** (mismo candidato `pickup_curve`, sin cambios de arquitectura): el benchmark y la producción usan la misma curva; los defaults se validan con un experimento A/B sobre la ventana de verificación (out-of-sample real) y el backtest T-90..T-1 por lead band.

## Capabilities

### New Capabilities

<!-- Ninguna. -->

### Modified Capabilities

- `data-ingestion`: ADDED — Propiedades cerradas excluidas: la exclusión configurada se aplica en la ingesta (tras alias) y se propaga a toda la corrida; el manifiesto la declara.
- `feature-engineering`: MODIFIED — Curvas de pickup no lineales: ahora exigidas condicionadas por temporada del stay con ventana de recencia y fallback por escasez (hace operativo el "por… mes, temporada" que la spec ya prometía).

## Impact

- **Código**: `src/config.py` + `config.yaml` (`excluded_properties`, `pickup_curve`), `src/models/baselines.py` (curva v2), `src/forecasting/core.py` (parametrización `curve_params` en fit/predict/build), `src/forecasting/runner.py` (filtro de exclusión + manifiesto), `src/forecasting/run_real.py` (exclusión en fuentes + parámetros al benchmark).
- **Comportamiento observable**: LIRAK/LAPAS desaparecen de todos los entregables (con registro explícito); la banda 30–89 del horizonte debe mejorar contra la corrida anterior; los warnings de calidad ligados a esas propiedades desaparecen.
- **Tests**: nuevos tests de exclusión y de la curva estacional (condicionamiento, recencia, fallback); suite completa en verde.
- **Riesgo**: el condicionamiento estacional podría empeorar leads cortos si un bucket es ruidoso → mitigado por `min_stays_per_bucket` con fallback y validación A/B por banda de lead antes de fijar defaults.
