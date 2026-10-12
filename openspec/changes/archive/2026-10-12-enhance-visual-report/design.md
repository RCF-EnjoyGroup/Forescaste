# Design — enhance-visual-report

## Context

`report.html` es la vista que el negocio consume (secciones de portafolio: forecast, forecast vs real, benchmark, pickup, ocupación, EDA). La operación es por propiedad — campeones por hotel desde la remediación — y las revisiones T-90..T-1 del backtest no aparecen. La vista es presentation-only (spec reporting: la lógica vive en `src/`).

## Goals / Non-Goals

**Goals** — Secciones por propiedad (forecast/bandas/modelo/reales, accuracy, revisiones con convergencia por lead band); defensivas; KPIs de campeón y cobertura; datos nuevos fluyen del runner sin tocar los 10 entregables (solo `backtesting_report.json` gana `by_property_lead_band`).

**Non-Goals** — Interactividad (JS/filtros — el reporte es estático autocontenido); charts de estacionalidad; Power BI (ya tiene el desglose por propiedad).

## Decisions

### D1 — Small multiples con eje independiente por propiedad
Las propiedades tienen escalas muy distintas (VILLAS ~30 RN vs CORIN ~200): un solo eje las haría ilegibles. Grid de mini-paneles (3 columnas), cada uno con P50 + banda P10/P90 + capacidad punteada + reales en rojo cuando hay; el título lleva `propiedad · modelo`. Alternativa descartada: un solo chart multiserie — spaghetti.

### D2 — Accuracy por propiedad: barras + tabla en la misma tarjeta
Las barras de WMAPE dan la lectura rápida y el ranking; la tabla (propiedad, modelo, WMAPE, MAPE, bias, n_obs) da los números exactos para revisión. Los datos vienen de `breakdowns["property"]` (ya calculado; se persiste en Power BI desde la remediación — la vista solo lo presenta).

### D3 — Revisiones: convergencia por banda de lead
La gráfica clave del backtest T-90..T-1 es cómo decae el error de cada propiedad conforme la estancia se acerca (eje: 90+ → 1-2). Para eso `_sampled_backtest` agrega `by_property_lead_band` (property × banda → n_obs, WMAPE, MAPE). Complemento: barras de WMAPE de revisiones por propiedad + tabla global por banda. La agregación vive en el runner (la vista no calcula métricas).

### D4 — Datos por kwargs opcionales, secciones defensivas
`generate_visual_report` gana `property_breakdown`, `model_by_property`, `backtest_report`, `champion` (todos keyword-only, default None): sin ellos el reporte queda idéntico al actual — el contrato "la vista jamás rompe la corrida" se preserva y cualquier caller antiguo sigue funcionando. KPIs: tarjeta Campeón (string) y Propiedades evaluadas `n/N` con formato especial.

## Risks / Trade-offs

- [HTML más pesado (más PNG base64)] — aceptado: sigue siendo un solo archivo portátil; dpi 110 y paneles compactos.
- [Muchas propiedades → grid alto] — grid auto (3 columnas); con 8 propiedades son 3 filas de mini-paneles.
- [Bandas de lead sin observaciones por propiedad] — la línea simplemente no tiene ese punto (NaN se salta).

## Migration Plan

1. `by_property_lead_band` en `_sampled_backtest` → secciones en `visual_report` → call site del runner → tests → corrida real → archive.
