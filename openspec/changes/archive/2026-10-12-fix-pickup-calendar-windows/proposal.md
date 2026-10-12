## Why

La base toma snapshots **diarios desde 2024**; antes, la cadencia era esparcida (medido: LIREL 2023 ≈ 5 snapshots por estancia vs ~150-200 desde 2024). Las features de pickup implementan "últimos N días" como "últimos N **snapshots**": en la era diaria ambas coinciden (por eso la producción no sufre daño medible), pero para estancias pre-2024 una ventana de "7 días" contenía hasta 7 semanas de booking. El spec feature-engineering exige "pickup de los últimos 1/3/7/14/30 **días**" — este cambio hace que la implementación cumpla el spec que ya existe (por eso `skip_specs: true`: la semántica requerida no cambia, la implementación converge a ella).

Contexto medido (verificación de que el MODELO ya opera en la era diaria): el pool de curvas con recencia 1095d contiene 0.0% de filas pre-2024 (15 de 628,206), el panel ML es 100% post-2024 (cap de 365 estancias) y el on-the-books de producción usa el snapshot exacto del corte (frescura 0 días). Este fix es corrección del frame de features, no del modelo.

## What Changes

- `pickup_last_{1,3,7,14,30}d`, `pickup_acceleration`, `pickup_deceleration` y `pickup_volatility` pasan a ventanas de **días calendario**: el nivel del on-the-books se reindexa a grilla diaria por (propiedad, estancia) con forward-fill entre snapshots, y las ventanas operan sobre los incrementos diarios. Con snapshots diarios el resultado es equivalente a la versión por snapshots; con cadencia esparcida (pre-2024) las ventanas cuentan días reales.
- Aceleración = pickup de los últimos 7 días − pickup de los 7 días previos (antes 3 snapshots vs 3 previos). Volatilidad = std de incrementos diarios en 7 días (antes 7 snapshots).
- `pickup_cumulative` se mantiene como está (acumulado por snapshot; el spec no define unidad) — documentado en docstring.
- Las features al mismo lead (`pickup_hist_*_same_lead`) no cambian (son por lead, inmunes a la cadencia).
- Prueba de neutralidad: la corrida real posterior debe reproducir **exactamente** las métricas de la corrida anterior (nada que consuma un modelo cambia).

## Capabilities

### New Capabilities

<!-- Ninguna. -->

### Modified Capabilities

<!-- Ninguna: el spec feature-engineering ya exige ventanas en días ("pickup de
los últimos 1/3/7/14/30 días"); este cambio hace que la implementación cumpla
el spec existente. skip_specs: true. -->

## Impact

- **Código**: `src/features/pickup.py` (ventanas a grilla diaria), `tests/test_features.py` (fórmulas day-based + test de regresión con cadencia esparcida + test de equivalencia en era diaria).
- **Comportamiento observable**: los valores de las features de ventana cambian SOLO para estancias pre-2024 del frame de features (ningún modelo las consume: `ML_FEATURES` no las incluye y el campeón usa curvas/requests); el forecast de producción debe quedar idéntico — verificado con corrida real.
- **Riesgo**: bajo y acotado; costo O(grilla diaria) por grupo (~150 días por estancia) absorbido en la fase de features.
