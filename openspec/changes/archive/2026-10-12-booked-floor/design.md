# Design — booked-floor-and-benchmark-viability

## Context

Verificación FIESTA: el campeón (sarimax) pronosticó por debajo del booked en 20/90 días (hasta −39 RN a lead 1), y el A/B mostró que pickup_curve gana en todas las bandas de la ventana de verificación pero perdió la selección porque los folds tempranos la puntuaron con ~24 días de historia (pickup de FIESTA empieza 2025-06-14).

## Goals / Non-Goals

**Goals** — Piso de booked como restricción operacional general; benchmark que juzgue candidatos donde fueron viables; curva en frío con umbral; verificación real de que FIESTA ya no publica forecasts bajo el booked.

**Non-Goals** — Modelo de cancelaciones (el piso no descuenta cancelaciones esperadas: la tasa empírica de FIESTA — 34/480 stays con booked final > real +10 RN — se documenta como trade-off); re-ponderar folds por recencia (rompería el tiling estacional); cambios por-hotel hardcodeados.

## Decisions

### D1 — El piso en la capa de restricciones, no en los modelos
`apply_capacity_constraint` gana el parámetro `booked_col` y computa `final = min(max(raw, booked), capacity)` — un solo lugar donde viven las invariantes (piso → techo, en ese orden) y cualquier campeón hereda la protección. `raw_forecast_rn` queda intacto para diagnóstico. El piso no aplica a la banda: las cancelaciones son un resultado posible (P10 < booked legítimo); solo el punto publicado y su p50 se pisan. Alternativa descartada: piso dentro de cada candidato — duplica lógica y los candidatos sin señal de booked no podrían.

### D2 — booked > capacidad: el techo manda + alerta
FIESTA tiene 49 snapshots con booked > 408. El piso no puede empujar el forecast por encima del inventario físico: en el conflicto gana la capacidad y se emite `booked_over_capacity` (inconsistencia de DATOS, ya reportada en el diagnóstico hotel por hotel — es la palanca para que Sistemas la corrija en la fuente).

### D3 — Umbral de curva en frío: 60 estancias
`_fit_pickup` degrada a nivel medio si `curves["stay_date"].nunique() < 60` (~6 estancias por bin con 10 bins — menos que eso es ruido ajustado como señal). FIESTA fold 1 (~24 estancias) queda en frío; fold 2 (~114) ya es viable. General y reversible por constante.

### D4 — Cold start no se puntúa; portafolio exige viabilidad completa
`real_fold_metrics` consulta `fitted.cold_start()` tras cada fit: candidato en frío → sin fila propiedad-fold, y sin fila de portafolio si no fue viable en TODAS las propiedades evaluadas. El fallback de nivel es una red de producción, no el candidato. La penalización actual castiga disponibilidad de datos, no calidad de modelo. El panel DM se construye sobre los (fold, fecha) comunes — la prueba ya lo maneja por construcción.

### D5 — El techo sigue evaluándose sin piso en los folds
El benchmark compara candidatos crudos (sin capa de restricciones) — el piso es post-hoc sobre el campeón. Sin cambios en el protocolo de evaluación de candidatos.

## Risks / Trade-offs

- [Piso conservador en fechas con cancelaciones] — documentado; medible (34/480 en FIESTA); un modelo de cancelaciones es el paso posterior natural.
- [Menos filas de fold para candidatos tardíos] — el composite usa los folds viables; la selección por propiedad exige ≥3 folds (FIESTA tendrá exactamente 3 para la curva).
- [Portafolio con menos filas para el panel DM] — 270 pares (3 folds) siguen siendo >3× la evidencia del protocolo original de 30d.
- [validate_forecast_table más estricto] — la fórmula floor+techo se verifica; tests existentes no rotan porque sus fixtures nunca pisan el piso (raw > booked).

## Migration Plan

Implementar → tests unitarios (piso, conflicto, curva fría, folds) → suite → corrida real → verificación FIESTA (campeón, 0 días bajo booked) → archive.
