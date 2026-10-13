# Design — lead-weighted-composite

## Decisions

### D1 — Split en lead 30, pesos 0.65/0.35
Los leads 1-30 concentran las decisiones operativas (staffing, pricing, walk-ins) y son donde un modelo plano causa daño visible (FIESTA: sarimax erra jueves por +80% a leads 15-30). Leads 31-89 son planeación — error más tolerable. El peso 0.65/0.35 significa que cada día del rango corto pesa ~2.8× el de largo (0.65/30 vs 0.35/60). Constantes: `LEAD_ALPHA = 0.65`, `LEAD_SPLIT = 30`.

### D2 — Desagregación en fold_metrics, no en selection
`real_fold_metrics` emite `wmape_short` y `wmape_long` por fila (model × fold × property). `score_candidates` las consume si existen; si no, cae al WMAPE plano. La desagregación vive donde ya están los datos (el loop de folds) — selection no necesita saber de leads.

### D3 — Estabilidad sin cambio
`stability = std(wmape)` sobre el WMAPE completo (no el ponderado) — mide consistencia entre folds, no calidad por lead. El bias penalty también sin cambio.

### D4 — DM sin cambio
La prueba de significancia compara campeón vs subcampeón sobre los mismos (fold, fecha) — es una pregunta estadística, no una ponderación operativa.

### D5 — Sin hardcode de FIESTA
El mecanismo es general: todas las propiedades obtienen el composite ponderado. Una propiedad sin swing de día-de-semana no cambia de campeón porque el WMAPE corto y largo serán similares.

## Risks / Trade-offs

- [Propiedad con buen largo, mal corto pierde la selección] — es el objetivo (un modelo que solo sirve a 60+ días no debe operar los 30 operativos). Se verifica contra regresión en TODAS las propiedades.
- [Corridas de muestra (sin las columnas nuevas) caen al WMAPE plano] — documentado; los tests existentes siguen pasando sin cambio.
