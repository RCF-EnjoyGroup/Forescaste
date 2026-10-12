# Design — fix-pickup-calendar-windows

## Decisions

### D1 — Grilla diaria con forward-fill del nivel, incrementos por día
Por (propiedad, estancia): serie `pickup_rn` indexada por `snapshot_date` (dedupe keep-last por fecha), reindexada a grilla diaria `min→max` con ffill. Los incrementos `inc = daily.diff()` valen 0 en días sin snapshot y concentran el salto el día del snapshot — así la suma sobre una ventana de días calendario es el pickup real de esos días, sin importar la cadencia. Las features se mapean de vuelta a cada fila por su `snapshot_date`.

### D2 — Ventanas day-based, parámetros equivalentes en la era diaria
`pickup_last_{w}d` = `inc.rolling(w).sum()`; aceleración = últimos 7 días − 7 días previos (`shift(7)` del rolling); volatilidad = std de `inc` en 7 días (`min_periods=2`). Con snapshots diarios estos valores replican (con mínimas diferencias en la primera fila) la versión por snapshots — test explícito de equivalencia.

### D3 — `pickup_cumulative` sin cambio
Su semántica de spec es ambigua ("acumulado") y su test fija el comportamiento actual; cambiarla no aporta nada a producción. Docstring declara que es por snapshot.

### D4 — Neutralidad de producción como criterio de aceptación
Como ninguna parte del modelo consume estas features (`ML_FEATURES` y las curvas no las usan), la corrida real posterior al fix debe reproducir las métricas de la corrida previa **exactamente** (MAPE 25.10% / WMAPE 0.1577 con folds 90d). Cualquier desviación en esas cifras invalidaría la premisa del cambio.
