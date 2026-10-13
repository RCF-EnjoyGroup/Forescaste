## Why

El archivo `capacity_daily.csv` exporta `business_date` en formato **DD/MM/YYYY** (estándar del negocio en Costa Rica), mientras que `demand_daily.csv` y `pickup_by_lead.csv` usan YYYY-MM-DD. El motor parsea todas las fechas con el default de pandas (que ante fechas con barras asume MM/DD), lo que hace que el **60.6% de las fechas de capacidad se vuelvan NaT y se descarten silenciosamente** (día > 12 no puede ser mes), y el 39% restante se interprete con el día y el mes cruzados (ej. "05/01/2025" = 5 de enero se lee como 5 de mayo). El modelo ha estado funcionando "por casualidad" porque casi todas las propiedades tienen capacidad constante (la mediana del fallback da el valor correcto y el techo rara vez constriñe), pero el matching por fecha está sistemáticamente equivocado: cualquier propiedad con capacidad cambiante recibiría el techo equivocado, y los checks de calidad muestrean solo el 39% de la serie.

Hallazgos de la investigación (2026-10-13):
- El perfil real de SJOSL solo es visible con fechas correctas: capacity=0 todo 2025 (propiedad cerrada), apertura gradual enero-junio 2026 (0→82). Con fechas cruzadas, esos cierres aparecían en fechas aleatorias.
- `check_rn_over_capacity` hoy solo cruza 3,499 de 8,851 filas de demanda (39%) y produce falsos positivos: LIREL 10 y SJOSL 7/108=6.5% por techos equivocados. Con el fix: 5 días reales de FIESTA (0.5% → warning, el gate pasa).
- **El capacity actual tiene 6 filas completamente vacías** (líneas ",," al final, edit accidental posterior a la corrida de las 09:32) que CRASHEAN la validación con `ValueError: Neither start nor end can be NaT` en `check_missing_dates`, y producirían falsos `duplicate_property_date` en error (NaN == NaN en duplicated). El próximo run fallaría sin hardening.

## What Changes

- **Parseo flexible de fechas**: nueva utilidad `parse_dates_flexible` con detección por patrón — ISO puro → formato estricto `%Y-%m-%d`; barras → evidencia día>12 decide DD/MM vs MM/DD; columna 100% ambigua → `prefer_dayfirst` desempata (solo events). Se aplica en la frontera de normalización: `_normalize_demand/_normalize_capacity/_normalize_pickup` (core), `SourceSchema.coerce` (schema), `_is_date_like` (detección), los checks de calidad que leen fechas crudas y el loader de `events.csv`. Nota de diseño: la comparación por conteo sola no sirve porque `dayfirst=True` corrompe fechas ISO ("2026-10-05" → 2026-05-10).
- **Higiene de filas vacías**: filas enteramente vacías (incluido el placeholder `'nan'` que deja el stringificado de alias) → nuevo check `empty_rows` (warning, se reportan y se filtran antes de los demás checks); grupos con fechas presentes pero ilegibles → issue de error sin crash (guarda por grupo en `check_missing_dates`).
- **El gate de la corrida real ve el capacity crudo**: `load_real_sources` devuelve también el capacity post alias/exclusión sin normalizar y `run_end_to_end` lo recibe — antes, el `dropna` de la normalización eliminaba la basura ANTES del gate y el reporte de producción nunca la mostraba.
- **La demanda y el pickup NO cambian de formato** — sus sources (ISO) ya parsean 100%; el fast-path los devuelve sin costo extra.
- **Sin cambios al gate**: la semántica de severidades queda igual; el fix hace que los checks operen sobre el 100% de la serie en vez del 39%.

## Capabilities

### Modified Capabilities
- `data-ingestion`: MODIFIED — Inspección automática de esquema: la detección/parseo de fechas maneja ISO y DD/MM/YYYY de forma transparente; validación de calidad robusta ante filas vacías y fechas ilegibles por grupo.

## Impact

- **Código**: `src/data/dates.py` (nuevo), `src/forecasting/core.py`, `src/data/schema.py`, `src/validation/quality.py`, `src/features/events.py`, `src/forecasting/run_real.py`.
- **Cableado**: `src/data/pipeline.py` y `runner.py` no cambian (usan las mismas funciones); `run_real.py` pasa el capacity crudo al gate.
- **Comportamiento observable**: el capacity pasa de 39% a ~100% de fechas parseadas; las capacidades se asocian a las FECHAS CORRECTAS de forma determinista (independiente del orden de filas del export); `check_rn_over_capacity` cruza 8,851 filas (vs 3,499) y elimina los falsos positivos de LIREL/SJOSL; el pipeline reporta filas vacías (`empty_rows`) sin crash.
- **Forense**: la corrida anterior (09:32) ya parseaba el capacity viejo ~100% POR ACCIDENTE — el orden de filas de ese export hacía que pandas infiriera DD/MM. El re-export de las 14:36 (que empieza con `01/01/2013` ambiguo) rompe el accidente: sin este fix, la próxima corrida usaría 39% de las fechas con día/mes cruzados. Con el fix, los deltas de métricas son exactamente 0.00.
- **Forecast**: el horizonte actual recibe capacidades constantes idénticas a las de hoy (54/408/139/64/81/222); features de capacidad y techos del backtest quedan mapeados a fechas correctas (CORIN 108→54, SJOSL apertura ene-2026, LIREL 85→147).
- **Riesgo**: bajo — las fuentes ISO toman el fast-path sin cambio; la rama DD/MM solo aplica ante evidencia día>12.
